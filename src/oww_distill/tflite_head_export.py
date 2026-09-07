from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float32)


def build_conv_attention_head(checkpoint: dict[str, Any]):
    import tensorflow as tf

    architecture = checkpoint["architecture"]
    if architecture["type"] != "conv_attention":
        raise ValueError(f"unsupported head architecture: {architecture['type']}")
    timesteps = int(architecture["n_timesteps"])
    embedding_dim = int(architecture["embedding_dim"])
    channels = int(architecture["layer_dim"])
    heads = int(architecture["n_heads"])
    head_dim = channels // heads
    state = checkpoint["model_state"]

    def constant(name: str, value) -> Any:
        return tf.constant(_numpy(value), dtype=tf.float32, name=name)

    feature_mean = constant("feature_mean", checkpoint["feature_mean"])
    feature_scale = constant("feature_scale", checkpoint["feature_scale"])
    conv0_weight = constant("conv0_weight", state["conv.0.weight"].permute(2, 1, 0))
    conv0_bias = constant("conv0_bias", state["conv.0.bias"])
    norm0_weight = constant("norm0_weight", state["conv.1.weight"])
    norm0_bias = constant("norm0_bias", state["conv.1.bias"])
    conv1_weight = constant("conv1_weight", state["conv.3.weight"].permute(2, 1, 0))
    conv1_bias = constant("conv1_bias", state["conv.3.bias"])
    norm1_weight = constant("norm1_weight", state["conv.4.weight"])
    norm1_bias = constant("norm1_bias", state["conv.4.bias"])
    qkv_weight = constant("qkv_weight", state["attention.in_proj_weight"].T)
    qkv_bias = constant("qkv_bias", state["attention.in_proj_bias"])
    attention_weight = constant("attention_weight", state["attention.out_proj.weight"].T)
    attention_bias = constant("attention_bias", state["attention.out_proj.bias"])
    attention_norm_weight = constant(
        "attention_norm_weight", state["attention_norm.weight"]
    )
    attention_norm_bias = constant("attention_norm_bias", state["attention_norm.bias"])
    output_weight = constant("output_weight", state["output.weight"].T)
    output_bias = constant("output_bias", state["output.bias"])

    def layer_norm_all(values, weight, bias):
        channel_first = tf.transpose(values, [0, 2, 1])
        mean, variance = tf.nn.moments(channel_first, axes=[1, 2], keepdims=True)
        normalized = (channel_first - mean) * tf.math.rsqrt(variance + 1.0e-5)
        return tf.transpose(normalized * weight[None] + bias[None], [0, 2, 1])

    def layer_norm_channels(values, weight, bias):
        mean, variance = tf.nn.moments(values, axes=[2], keepdims=True)
        return (values - mean) * tf.math.rsqrt(variance + 1.0e-5) * weight + bias

    class ConvAttentionModule(tf.Module):
        @tf.function
        def infer(self, embeddings):
            hidden = (embeddings - feature_mean) / feature_scale
            hidden = tf.nn.conv1d(hidden, conv0_weight, stride=1, padding="SAME")
            hidden = tf.nn.bias_add(hidden, conv0_bias)
            hidden = tf.nn.relu(layer_norm_all(hidden, norm0_weight, norm0_bias))
            hidden = tf.nn.conv1d(hidden, conv1_weight, stride=1, padding="SAME")
            hidden = tf.nn.bias_add(hidden, conv1_bias)
            hidden = tf.nn.relu(layer_norm_all(hidden, norm1_weight, norm1_bias))

            qkv = tf.matmul(hidden, qkv_weight) + qkv_bias
            query, key, value = tf.split(qkv, 3, axis=-1)

            def split_heads(values):
                values = tf.reshape(values, [1, timesteps, heads, head_dim])
                return tf.transpose(values, [0, 2, 1, 3])

            query = split_heads(query)
            key = split_heads(key)
            value = split_heads(value)
            logits = tf.matmul(query, key, transpose_b=True) / np.sqrt(float(head_dim))
            attended = tf.matmul(tf.nn.softmax(logits, axis=-1), value)
            attended = tf.reshape(
                tf.transpose(attended, [0, 2, 1, 3]), [1, timesteps, channels]
            )
            attended = tf.matmul(attended, attention_weight) + attention_bias
            hidden = layer_norm_channels(
                hidden + attended, attention_norm_weight, attention_norm_bias
            )
            pooled = tf.reduce_mean(hidden, axis=1)
            logit = tf.squeeze(tf.matmul(pooled, output_weight) + output_bias, axis=1)
            return {"score": tf.sigmoid(logit)}

    module = ConvAttentionModule()
    concrete = module.infer.get_concrete_function(
        tf.TensorSpec([1, timesteps, embedding_dim], tf.float32, name="embeddings")
    )
    return module, concrete


def convert_head(concrete, module, representative_embeddings, output: Path) -> bytes:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], module)
    if representative_embeddings is not None:
        values = np.asarray(representative_embeddings, dtype=np.float32)

        def representative_dataset():
            for embeddings in values:
                yield {"embeddings": embeddings[None]}

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    model = converter.convert()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(model)
    return model


def quantize(value: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    scale, zero_point = detail["quantization"]
    bounds = np.iinfo(detail["dtype"])
    return np.clip(np.rint(value / scale) + zero_point, bounds.min, bounds.max).astype(
        detail["dtype"]
    )


class TFLiteWakeHeadRuntime:
    def __init__(
        self, model_path: Path, threshold: float = 0.0, name: str = "hey_pico"
    ) -> None:
        from ai_edge_litert.interpreter import Interpreter

        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.runner = self.interpreter.get_signature_runner("serving_default")
        self.input = self.runner.get_input_details()["embeddings"]
        self.output = self.runner.get_output_details()["score"]
        self.threshold = threshold
        self.name = name

    def scores(self, embeddings: np.ndarray) -> np.ndarray:
        values = np.asarray(embeddings, dtype=np.float32)
        result = []
        for value in values:
            output = self.runner(embeddings=quantize(value[None], self.input))["score"]
            scale, zero_point = self.output["quantization"]
            result.append((output.astype(np.float32) - zero_point) * scale)
        return np.concatenate(result)

    def score(self, embeddings: np.ndarray) -> float:
        return float(self.scores(np.asarray(embeddings)[None])[0])
