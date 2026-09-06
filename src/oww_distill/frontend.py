from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16_000
FFT_SIZE = 512
WINDOW_SAMPLES = 400
HOP_SAMPLES = 160
MEL_BINS = 32


@dataclass(frozen=True)
class FrontendConstants:
    window: np.ndarray
    mel_weights: np.ndarray

    def __post_init__(self) -> None:
        if self.window.shape != (FFT_SIZE,):
            raise ValueError(f"window must have shape {(FFT_SIZE,)}, got {self.window.shape}")
        if self.mel_weights.shape != (FFT_SIZE // 2 + 1, MEL_BINS):
            raise ValueError(
                "mel_weights must have shape "
                f"{(FFT_SIZE // 2 + 1, MEL_BINS)}, got {self.mel_weights.shape}"
            )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            format_version=np.array(1, dtype=np.int32),
            sample_rate=np.array(SAMPLE_RATE, dtype=np.int32),
            fft_size=np.array(FFT_SIZE, dtype=np.int32),
            window_samples=np.array(WINDOW_SAMPLES, dtype=np.int32),
            hop_samples=np.array(HOP_SAMPLES, dtype=np.int32),
            window=self.window.astype(np.float32),
            mel_weights=self.mel_weights.astype(np.float32),
        )

    @classmethod
    def load(cls, path: Path) -> "FrontendConstants":
        with np.load(path, allow_pickle=False) as values:
            if int(values["format_version"]) != 1:
                raise ValueError("unsupported frontend constants format")
            return cls(values["window"].copy(), values["mel_weights"].copy())


def extract_tflite_constants(model_path: Path) -> FrontendConstants:
    """Extract the fixed Hann/DFT envelope and Mel matrix from openWakeWord."""
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(model_path))
    interpreter.resize_tensor_input(0, [1, 1280], strict=True)
    interpreter.allocate_tensors()
    tensors = interpreter.get_tensor_details()
    by_shape: dict[tuple[int, ...], list[int]] = {}
    for tensor in tensors:
        by_shape.setdefault(tuple(int(v) for v in tensor["shape"]), []).append(tensor["index"])
    dft_indices = by_shape.get((257, 1, 512, 1), [])
    mel_indices = by_shape.get((257, 32), [])
    if len(dft_indices) != 2 or len(mel_indices) != 1:
        raise RuntimeError("unexpected openWakeWord melspectrogram graph")
    real = interpreter.get_tensor(dft_indices[0])[:, 0, :, 0]
    imag = interpreter.get_tensor(dft_indices[1])[:, 0, :, 0]
    # Magnitude of any non-degenerate DFT row recovers the embedded Hann window.
    window = np.sqrt(real[10] ** 2 + imag[10] ** 2)
    mel_weights = interpreter.get_tensor(mel_indices[0])
    return FrontendConstants(window.astype(np.float32), mel_weights.astype(np.float32))


class DspFrontend:
    """NumPy RFFT equivalent of openWakeWord's float TFLite frontend."""

    def __init__(self, constants: FrontendConstants) -> None:
        self.constants = constants

    def __call__(self, pcm: np.ndarray) -> np.ndarray:
        pcm = np.asarray(pcm)
        if pcm.ndim != 1:
            raise ValueError("PCM must be one-dimensional")
        if pcm.size < FFT_SIZE:
            return np.empty((0, MEL_BINS), dtype=np.float32)
        x = pcm.astype(np.float32, copy=False)
        frames = np.lib.stride_tricks.sliding_window_view(x, FFT_SIZE)[::HOP_SAMPLES]
        spectrum = np.fft.rfft(frames * self.constants.window, axis=1)
        power = spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
        energy = power @ self.constants.mel_weights
        db = 10.0 * np.log10(np.maximum(energy, 1.0e-10))
        # The reference graph reduces over the complete invocation, not each
        # frame independently. This matters for impulses and quiet frames.
        db = np.maximum(db, db.max() - 80.0)
        return (db / 10.0 + 2.0).astype(np.float32)

    def streaming_clip(self, pcm: np.ndarray, chunk_samples: int = 1280) -> np.ndarray:
        """Simulate openWakeWord's 80-ms frontend calls with 30-ms overlap."""
        pcm = np.asarray(pcm)
        chunks: list[np.ndarray] = []
        for end in range(chunk_samples, pcm.size + 1, chunk_samples):
            start = max(0, end - chunk_samples - 3 * HOP_SAMPLES)
            chunks.append(self(pcm[start:end]))
        remainder = pcm.size % chunk_samples
        if remainder:
            end = pcm.size
            start = max(0, end - remainder - 3 * HOP_SAMPLES)
            chunks.append(self(pcm[start:end]))
        return np.concatenate(chunks) if chunks else np.empty((0, MEL_BINS), np.float32)
