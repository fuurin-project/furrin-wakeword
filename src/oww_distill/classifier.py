from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


MODEL_FORMAT_VERSION = 1


@dataclass(frozen=True)
class LinearCommandClassifier:
    weights: np.ndarray
    bias: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    classes: np.ndarray

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float32)
        bias = np.asarray(self.bias, dtype=np.float32).reshape(-1)
        feature_mean = np.asarray(self.feature_mean, dtype=np.float32).reshape(-1)
        feature_scale = np.asarray(self.feature_scale, dtype=np.float32).reshape(-1)
        classes = np.asarray(self.classes).astype(str).reshape(-1)
        if weights.ndim != 2:
            raise ValueError("weights must be a class-by-feature matrix")
        if weights.shape != (len(classes), len(feature_mean)):
            raise ValueError("weights, classes, and feature mean shapes do not match")
        if bias.shape != (len(classes),) or feature_scale.shape != feature_mean.shape:
            raise ValueError("bias or feature scale shape does not match")
        if np.any(feature_scale <= 0):
            raise ValueError("feature scale must be positive")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "feature_mean", feature_mean)
        object.__setattr__(self, "feature_scale", feature_scale)
        object.__setattr__(self, "classes", classes)

    def probabilities(self, embedding: np.ndarray) -> np.ndarray:
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if embedding.shape != self.feature_mean.shape:
            raise ValueError(
                f"embedding must have shape {self.feature_mean.shape}, got {embedding.shape}"
            )
        normalized = (embedding - self.feature_mean) / self.feature_scale
        logits = self.weights @ normalized + self.bias
        logits -= logits.max()
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            format_version=np.array(MODEL_FORMAT_VERSION, dtype=np.int32),
            weights=self.weights,
            bias=self.bias,
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            classes=self.classes,
            pooling=np.array("temporal_mean"),
        )

    @classmethod
    def load(cls, path: Path) -> "LinearCommandClassifier":
        with np.load(path, allow_pickle=False) as values:
            if int(values["format_version"]) != MODEL_FORMAT_VERSION:
                raise ValueError("unsupported command classifier format")
            if str(values["pooling"]) != "temporal_mean":
                raise ValueError("unsupported command classifier pooling")
            return cls(
                weights=values["weights"],
                bias=values["bias"],
                feature_mean=values["feature_mean"],
                feature_scale=values["feature_scale"],
                classes=values["classes"],
            )


@dataclass(frozen=True)
class BinaryWakeClassifier:
    hidden_weights: np.ndarray
    hidden_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    threshold: float
    name: str = "hey_pico"

    def __post_init__(self) -> None:
        hidden_weights = np.asarray(self.hidden_weights, dtype=np.float32)
        hidden_bias = np.asarray(self.hidden_bias, dtype=np.float32).reshape(-1)
        output_weights = np.asarray(self.output_weights, dtype=np.float32).reshape(-1)
        feature_mean = np.asarray(self.feature_mean, dtype=np.float32).reshape(-1)
        feature_scale = np.asarray(self.feature_scale, dtype=np.float32).reshape(-1)
        if hidden_weights.shape != (len(feature_mean), len(hidden_bias)):
            raise ValueError("hidden layer shape does not match feature statistics")
        if output_weights.shape != hidden_bias.shape:
            raise ValueError("output layer shape does not match hidden layer")
        if feature_scale.shape != feature_mean.shape or np.any(feature_scale <= 0):
            raise ValueError("invalid feature scale")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between zero and one")
        object.__setattr__(self, "hidden_weights", hidden_weights)
        object.__setattr__(self, "hidden_bias", hidden_bias)
        object.__setattr__(self, "output_weights", output_weights)
        object.__setattr__(self, "feature_mean", feature_mean)
        object.__setattr__(self, "feature_scale", feature_scale)

    def score(self, embeddings: np.ndarray) -> float:
        features = np.asarray(embeddings, dtype=np.float32).reshape(-1)
        if features.shape != self.feature_mean.shape:
            raise ValueError(
                f"embedding must flatten to {self.feature_mean.shape}, got {features.shape}"
            )
        normalized = (features - self.feature_mean) / self.feature_scale
        hidden = np.maximum(normalized @ self.hidden_weights + self.hidden_bias, 0.0)
        logit = float(hidden @ self.output_weights + self.output_bias)
        if logit >= 0:
            return float(1.0 / (1.0 + np.exp(-logit)))
        exponential = np.exp(logit)
        return float(exponential / (1.0 + exponential))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            format_version=np.array(MODEL_FORMAT_VERSION, dtype=np.int32),
            hidden_weights=self.hidden_weights,
            hidden_bias=self.hidden_bias,
            output_weights=self.output_weights,
            output_bias=np.array(self.output_bias, dtype=np.float32),
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            threshold=np.array(self.threshold, dtype=np.float32),
            name=np.array(self.name),
            pooling=np.array("flatten_16x96"),
        )

    @classmethod
    def load(cls, path: Path) -> "BinaryWakeClassifier":
        with np.load(path, allow_pickle=False) as values:
            if int(values["format_version"]) != MODEL_FORMAT_VERSION:
                raise ValueError("unsupported wake classifier format")
            if str(values["pooling"]) != "flatten_16x96":
                raise ValueError("unsupported wake classifier pooling")
            return cls(
                hidden_weights=values["hidden_weights"],
                hidden_bias=values["hidden_bias"],
                output_weights=values["output_weights"],
                output_bias=float(values["output_bias"]),
                feature_mean=values["feature_mean"],
                feature_scale=values["feature_scale"],
                threshold=float(values["threshold"]),
                name=str(values["name"]),
            )


@dataclass(frozen=True)
class TemporalWakeClassifier:
    depthwise_weights: np.ndarray
    depthwise_bias: np.ndarray
    pointwise_weights: np.ndarray
    pointwise_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    threshold: float
    name: str = "hey_pico"

    def __post_init__(self) -> None:
        depthwise_weights = np.asarray(self.depthwise_weights, dtype=np.float32)
        depthwise_bias = np.asarray(self.depthwise_bias, dtype=np.float32).reshape(-1)
        pointwise_weights = np.asarray(self.pointwise_weights, dtype=np.float32)
        pointwise_bias = np.asarray(self.pointwise_bias, dtype=np.float32).reshape(-1)
        output_weights = np.asarray(self.output_weights, dtype=np.float32).reshape(-1)
        feature_mean = np.asarray(self.feature_mean, dtype=np.float32).reshape(-1)
        feature_scale = np.asarray(self.feature_scale, dtype=np.float32).reshape(-1)
        if depthwise_weights.ndim != 2 or depthwise_weights.shape[0] != len(feature_mean):
            raise ValueError("depthwise weights must be feature-by-kernel")
        if depthwise_weights.shape[1] % 2 != 1:
            raise ValueError("depthwise kernel size must be odd")
        if depthwise_bias.shape != feature_mean.shape or feature_scale.shape != feature_mean.shape:
            raise ValueError("depthwise bias or feature statistics shape mismatch")
        if pointwise_weights.ndim != 2 or pointwise_weights.shape[1] != len(feature_mean):
            raise ValueError("pointwise weights must be channel-by-feature")
        if pointwise_bias.shape != (pointwise_weights.shape[0],):
            raise ValueError("pointwise bias shape mismatch")
        if output_weights.shape != (2 * pointwise_weights.shape[0],):
            raise ValueError("output weights must consume max and mean pooled channels")
        if np.any(feature_scale <= 0) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("invalid feature scale or threshold")
        object.__setattr__(self, "depthwise_weights", depthwise_weights)
        object.__setattr__(self, "depthwise_bias", depthwise_bias)
        object.__setattr__(self, "pointwise_weights", pointwise_weights)
        object.__setattr__(self, "pointwise_bias", pointwise_bias)
        object.__setattr__(self, "output_weights", output_weights)
        object.__setattr__(self, "feature_mean", feature_mean)
        object.__setattr__(self, "feature_scale", feature_scale)

    def score(self, embeddings: np.ndarray) -> float:
        frames = np.asarray(embeddings, dtype=np.float32)
        if frames.ndim != 2 or frames.shape[1] != len(self.feature_mean):
            raise ValueError(
                f"embeddings must have shape (frames, {len(self.feature_mean)}), got {frames.shape}"
            )
        frames = (frames - self.feature_mean) / self.feature_scale
        kernel_size = self.depthwise_weights.shape[1]
        padding = kernel_size // 2
        padded = np.pad(frames, ((padding, padding), (0, 0)))
        depthwise = np.zeros_like(frames)
        for offset in range(kernel_size):
            depthwise += padded[offset : offset + len(frames)] * self.depthwise_weights[:, offset]
        depthwise += self.depthwise_bias
        hidden = np.maximum(depthwise @ self.pointwise_weights.T + self.pointwise_bias, 0.0)
        pooled = np.concatenate((hidden.max(axis=0), hidden.mean(axis=0)))
        logit = float(pooled @ self.output_weights + self.output_bias)
        if logit >= 0:
            return float(1.0 / (1.0 + np.exp(-logit)))
        exponential = np.exp(logit)
        return float(exponential / (1.0 + exponential))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            format_version=np.array(MODEL_FORMAT_VERSION, dtype=np.int32),
            depthwise_weights=self.depthwise_weights,
            depthwise_bias=self.depthwise_bias,
            pointwise_weights=self.pointwise_weights,
            pointwise_bias=self.pointwise_bias,
            output_weights=self.output_weights,
            output_bias=np.array(self.output_bias, dtype=np.float32),
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            threshold=np.array(self.threshold, dtype=np.float32),
            name=np.array(self.name),
            pooling=np.array("temporal_depthwise_max_mean"),
        )

    @classmethod
    def load(cls, path: Path) -> "TemporalWakeClassifier":
        with np.load(path, allow_pickle=False) as values:
            if int(values["format_version"]) != MODEL_FORMAT_VERSION:
                raise ValueError("unsupported wake classifier format")
            if str(values["pooling"]) != "temporal_depthwise_max_mean":
                raise ValueError("unsupported temporal wake classifier pooling")
            return cls(
                depthwise_weights=values["depthwise_weights"],
                depthwise_bias=values["depthwise_bias"],
                pointwise_weights=values["pointwise_weights"],
                pointwise_bias=values["pointwise_bias"],
                output_weights=values["output_weights"],
                output_bias=float(values["output_bias"]),
                feature_mean=values["feature_mean"],
                feature_scale=values["feature_scale"],
                threshold=float(values["threshold"]),
                name=str(values["name"]),
            )


class TorchConvAttentionWakeClassifier:
    """Host runtime for a trained Conv-Attention head checkpoint."""

    def __init__(self, path: Path, device: str = "cpu") -> None:
        import torch

        from oww_distill.conv_attention import ConvAttentionHead

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        self._torch = torch
        self.device = torch.device(device)
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        architecture = checkpoint["architecture"]
        if architecture["type"] != "conv_attention":
            raise ValueError(f"unsupported torch wake head: {architecture['type']}")
        self.model = ConvAttentionHead(
            n_timesteps=int(architecture["n_timesteps"]),
            embedding_dim=int(architecture["embedding_dim"]),
            layer_dim=int(architecture["layer_dim"]),
            n_blocks=int(architecture["n_blocks"]),
            n_heads=int(architecture["n_heads"]),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.feature_mean = checkpoint["feature_mean"].to(self.device).reshape(1, 1, -1)
        self.feature_scale = checkpoint["feature_scale"].to(self.device).reshape(1, 1, -1)
        self.threshold = float(checkpoint["threshold"])
        self.name = str(checkpoint.get("name", "hey_pico"))

    def score(self, embeddings: np.ndarray) -> float:
        frames = np.asarray(embeddings, dtype=np.float32)
        expected = (self.model.n_timesteps, self.model.embedding_dim)
        if frames.shape != expected:
            raise ValueError(f"embeddings must have shape {expected}, got {frames.shape}")
        tensor = self._torch.from_numpy(frames).to(self.device).reshape(1, *expected)
        tensor = (tensor - self.feature_mean) / self.feature_scale
        with self._torch.inference_mode():
            return float(self._torch.sigmoid(self.model(tensor))[0].cpu())


def load_wake_classifier(
    path: Path, device: str = "cpu"
) -> BinaryWakeClassifier | TemporalWakeClassifier | TorchConvAttentionWakeClassifier:
    if path.suffix == ".pt":
        return TorchConvAttentionWakeClassifier(path, device)
    with np.load(path, allow_pickle=False) as values:
        pooling = str(values["pooling"])
    if pooling == "flatten_16x96":
        return BinaryWakeClassifier.load(path)
    if pooling == "temporal_depthwise_max_mean":
        return TemporalWakeClassifier.load(path)
    raise ValueError(f"unsupported wake classifier pooling: {pooling}")


class DistilledStudentRuntime:
    def __init__(self, checkpoint_path: Path, constants_path: Path, device: str = "cpu") -> None:
        import torch

        from oww_distill.frontend import DspFrontend, FrontendConstants
        from oww_distill.model import TinyEmbeddingStudent

        self._torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        architecture = checkpoint["architecture"]
        self.model = TinyEmbeddingStudent(
            feature_bins=int(architecture["feature_bins"]),
            channels=int(architecture["channels"]),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.frontend = DspFrontend(FrontendConstants.load(constants_path))
        self.input_scale = float(checkpoint["input_scale"])
        self.input_zero_point = int(checkpoint["input_zero_point"])
        self.feature_mean = checkpoint["feature_mean"].to(self.device).reshape(1, 1, -1)
        self.feature_std = checkpoint["feature_std"].to(self.device).reshape(1, 1, -1)

    @staticmethod
    def center_pad_or_trim(pcm: np.ndarray, samples: int = 32_000) -> np.ndarray:
        pcm = np.asarray(pcm, dtype=np.int16).reshape(-1)
        if len(pcm) >= samples:
            start = (len(pcm) - samples) // 2
            return pcm[start : start + samples].copy()
        output = np.zeros(samples, dtype=np.int16)
        start = (samples - len(pcm)) // 2
        output[start : start + len(pcm)] = pcm
        return output

    def features_pcm(self, pcm: np.ndarray) -> np.ndarray:
        pcm = self.center_pad_or_trim(pcm)
        features = self.frontend.streaming_clip(pcm)
        if features.shape != (197, 32):
            raise RuntimeError(f"unexpected frontend shape: {features.shape}")
        return features

    def embed_feature_batch(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 3 or features.shape[1:] != (197, 32):
            raise ValueError(f"feature batch must have shape (N, 197, 32), got {features.shape}")
        quantized = np.clip(
            np.rint(features / self.input_scale) + self.input_zero_point, -128, 127
        )
        dequantized = (quantized - self.input_zero_point) * self.input_scale
        tensor = self._torch.from_numpy(dequantized.astype(np.float32)).to(self.device)
        tensor = (tensor - self.feature_mean) / self.feature_std
        with self._torch.inference_mode():
            return self.model(tensor).cpu().numpy()

    def embed_frames_pcm(self, pcm: np.ndarray) -> np.ndarray:
        return self.embed_feature_batch(self.features_pcm(pcm)[None])[0]

    def embed_pcm(self, pcm: np.ndarray) -> np.ndarray:
        return self.embed_frames_pcm(pcm).mean(axis=0)
