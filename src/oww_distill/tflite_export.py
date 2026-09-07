from __future__ import annotations

from collections.abc import Iterator
import hashlib
from pathlib import Path
from typing import Any

import numpy as np


DILATIONS = (1, 2, 4, 8, 16, 32)


def load_student_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def quantized_features(features: np.ndarray, checkpoint: dict[str, Any]) -> np.ndarray:
    """Apply the frozen raw-feature quantizer used during student training."""
    values = np.asarray(features, dtype=np.float32)
    scale = float(checkpoint["input_scale"])
    zero_point = int(checkpoint["input_zero_point"])
    quantized = np.clip(np.rint(values / scale) + zero_point, -128, 127)
    return ((quantized - zero_point) * scale).astype(np.float32)


def normalize_features(features: np.ndarray, checkpoint: dict[str, Any]) -> np.ndarray:
    """Apply the frozen affine input quantizer and training normalization."""
    dequantized = quantized_features(features, checkpoint)
    mean = checkpoint["feature_mean"].cpu().numpy().reshape(1, 1, -1)
    std = checkpoint["feature_std"].cpu().numpy().reshape(1, 1, -1)
    return ((dequantized - mean) / std).astype(np.float32)


def zero_states(channels: int = 64) -> list[np.ndarray]:
    return [np.zeros((1, 2 * dilation, channels), np.float32) for dilation in DILATIONS]


def streaming_aligned_features(path: Path, label: int, kind: str, frontend) -> np.ndarray:
    """Reproduce the deterministic alignment used by the streaming head cache."""
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="int16", always_2d=False)
    if audio.ndim == 2:
        audio = np.rint(audio.astype(np.float32).mean(axis=1)).astype(np.int16)
    audio = np.asarray(audio, dtype=np.int16).reshape(-1)
    if sample_rate != 16_000:
        positions = np.linspace(
            0, len(audio) - 1,
            max(1, int(round(len(audio) * 16_000 / sample_rate))),
        )
        audio = np.interp(positions, np.arange(len(audio)), audio).astype(np.int16)
    window = np.zeros(32_000, dtype=np.int16)
    align_to_end = label == 1 or kind != "noise"
    if align_to_end:
        seed = int.from_bytes(hashlib.sha256(str(path).encode()).digest()[:8], "little")
        jitter = seed % 3201
        end = 32_000 - jitter
        start = max(0, end - len(audio))
        source_start = max(0, len(audio) - (end - start))
        window[start:end] = audio[source_start : source_start + end - start]
    elif len(audio) >= 32_000:
        start = (len(audio) - 32_000) // 2
        window = audio[start : start + 32_000].copy()
    else:
        start = (32_000 - len(audio)) // 2
        window[start : start + len(audio)] = audio
    return frontend.streaming_clip(window)


def build_streaming_student(checkpoint: dict[str, Any]):
    """Build a one-Mel-frame TensorFlow module with explicit causal states."""
    import tensorflow as tf

    state_dict = checkpoint["model_state"]
    architecture = checkpoint["architecture"]
    channels = int(architecture["channels"])
    feature_bins = int(architecture["feature_bins"])
    embedding_dim = int(architecture["embedding_dim"])

    def variable(name: str, value: np.ndarray):
        return tf.Variable(value.astype(np.float32), trainable=False, name=name)

    class StreamingStudent(tf.Module):
        def __init__(self) -> None:
            super().__init__(name="fuurin_streaming_student")
            value = state_dict["input_projection.weight"].cpu().numpy()
            mean = checkpoint["feature_mean"].cpu().numpy().reshape(-1)
            std = checkpoint["feature_std"].cpu().numpy().reshape(-1)
            input_bias = state_dict["input_projection.bias"].cpu().numpy()
            folded_value = value / std[None, :, None]
            folded_bias = input_bias - np.sum(value[:, :, 0] * mean[None, :] / std, axis=1)
            self.input_kernel = variable(
                "input_kernel", folded_value.transpose(2, 1, 0)[None, ...]
            )
            self.input_bias = variable("input_bias", folded_bias)
            self.depthwise_kernels = []
            self.depthwise_biases = []
            self.pointwise_kernels = []
            self.pointwise_biases = []
            for index in range(len(DILATIONS)):
                prefix = f"blocks.{index}"
                depthwise = state_dict[f"{prefix}.depthwise.weight"].cpu().numpy()
                pointwise = state_dict[f"{prefix}.pointwise.weight"].cpu().numpy()
                self.depthwise_kernels.append(
                    variable(f"depthwise_kernel_{index}", depthwise.transpose(2, 0, 1)[None])
                )
                self.depthwise_biases.append(
                    variable(
                        f"depthwise_bias_{index}",
                        state_dict[f"{prefix}.depthwise.bias"].cpu().numpy(),
                    )
                )
                self.pointwise_kernels.append(
                    variable(
                        f"pointwise_kernel_{index}", pointwise.transpose(2, 1, 0)[None, ...]
                    )
                )
                self.pointwise_biases.append(
                    variable(
                        f"pointwise_bias_{index}",
                        state_dict[f"{prefix}.pointwise.bias"].cpu().numpy(),
                    )
                )
            output = state_dict["output_projection.weight"].cpu().numpy()
            self.output_kernel = variable(
                "output_kernel", output.transpose(2, 1, 0)[None, ...]
            )
            self.output_bias = variable(
                "output_bias", state_dict["output_projection.bias"].cpu().numpy()
            )

        @tf.function
        def infer(
            self,
            feature,
            state_0,
            state_1,
            state_2,
            state_3,
            state_4,
            state_5,
        ):
            states = (state_0, state_1, state_2, state_3, state_4, state_5)
            hidden = tf.expand_dims(feature, axis=1)
            hidden = tf.nn.conv2d(hidden, self.input_kernel, strides=1, padding="VALID")
            hidden = tf.nn.relu(tf.nn.bias_add(hidden, self.input_bias))
            next_states = []
            for index, (dilation, state) in enumerate(zip(DILATIONS, states, strict=True)):
                residual = hidden
                context = tf.concat([tf.expand_dims(state, axis=1), residual], axis=2)
                hidden = tf.nn.depthwise_conv2d(
                    context,
                    self.depthwise_kernels[index],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    dilations=[1, dilation],
                )
                hidden = tf.nn.bias_add(hidden, self.depthwise_biases[index])
                hidden = tf.nn.conv2d(
                    hidden, self.pointwise_kernels[index], strides=1, padding="VALID"
                )
                hidden = tf.nn.relu(
                    tf.nn.bias_add(hidden, self.pointwise_biases[index]) + residual
                )
                next_states.append(tf.squeeze(context[:, :, 1:, :], axis=1))
            embedding = tf.nn.conv2d(
                hidden, self.output_kernel, strides=1, padding="VALID"
            )
            embedding = tf.nn.bias_add(embedding, self.output_bias)
            return {
                "embedding": tf.squeeze(embedding, axis=1),
                **{f"next_state_{i}": value for i, value in enumerate(next_states)},
            }

    module = StreamingStudent()
    input_specs = [
        tf.TensorSpec([1, 1, feature_bins], tf.float32, name="feature"),
        *[
            tf.TensorSpec([1, 2 * dilation, channels], tf.float32, name=f"state_{index}")
            for index, dilation in enumerate(DILATIONS)
        ],
    ]
    concrete = module.infer.get_concrete_function(*input_specs)
    return module, concrete


def representative_steps(
    normalized_features: np.ndarray,
    module,
    sample_stride: int = 8,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield real feature/state pairs so every recurrent tensor is calibrated."""
    import tensorflow as tf

    for clip in normalized_features:
        states = zero_states()
        for frame_index, frame in enumerate(clip):
            inputs = {
                "feature": frame.reshape(1, 1, -1).astype(np.float32),
                **{f"state_{i}": state for i, state in enumerate(states)},
            }
            outputs = module.infer(
                tf.convert_to_tensor(inputs["feature"]),
                *[tf.convert_to_tensor(inputs[f"state_{i}"]) for i in range(len(DILATIONS))],
            )
            if frame_index % sample_stride == 0:
                yield inputs
            states = [outputs[f"next_state_{i}"].numpy() for i in range(len(DILATIONS))]


def convert_streaming_student(
    concrete,
    module,
    output: Path,
    representative_dataset=None,
) -> bytes:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], module)
    if representative_dataset is not None:
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
    dtype = detail["dtype"]
    if np.issubdtype(dtype, np.floating):
        return value.astype(dtype)
    scale, zero_point = detail["quantization"]
    if scale <= 0:
        raise ValueError(f"tensor {detail['name']} has no quantization scale")
    bounds = np.iinfo(dtype)
    return np.clip(np.rint(value / scale) + zero_point, bounds.min, bounds.max).astype(dtype)


def dequantize(value: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    if np.issubdtype(detail["dtype"], np.floating):
        return value.astype(np.float32)
    scale, zero_point = detail["quantization"]
    return (value.astype(np.float32) - zero_point) * scale


class StreamingTFLiteRuntime:
    def __init__(self, model_path: Path) -> None:
        from ai_edge_litert.interpreter import Interpreter

        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.runner = self.interpreter.get_signature_runner("serving_default")
        self.inputs = self.runner.get_input_details()
        self.outputs = self.runner.get_output_details()
        expected_inputs = {"feature", *(f"state_{i}" for i in range(len(DILATIONS)))}
        expected_outputs = {"embedding", *(f"next_state_{i}" for i in range(len(DILATIONS)))}
        if self.inputs.keys() != expected_inputs or self.outputs.keys() != expected_outputs:
            raise ValueError(
                f"unexpected TFLite signature: inputs={self.inputs.keys()}, "
                f"outputs={self.outputs.keys()}"
            )
        self.reset()

    def reset(self) -> None:
        self.states = zero_states()

    def step(self, feature: np.ndarray) -> np.ndarray:
        values = {
            "feature": np.asarray(feature, np.float32).reshape(1, 1, 32),
            **{f"state_{i}": state for i, state in enumerate(self.states)},
        }
        outputs = self.runner(
            **{key: quantize(values[key], detail) for key, detail in self.inputs.items()}
        )
        embedding = dequantize(outputs["embedding"], self.outputs["embedding"])
        self.states = [
            dequantize(outputs[f"next_state_{i}"], self.outputs[f"next_state_{i}"])
            for i in range(len(DILATIONS))
        ]
        return embedding.reshape(96)

    def frame_embeddings(self, features: np.ndarray) -> np.ndarray:
        self.reset()
        return np.stack([self.step(frame) for frame in np.asarray(features, np.float32)])

    def selected_embeddings(self, features: np.ndarray) -> np.ndarray:
        from oww_distill.model import TEACHER_INDICES

        return self.frame_embeddings(features)[list(TEACHER_INDICES)]
