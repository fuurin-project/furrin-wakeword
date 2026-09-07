from pathlib import Path

import numpy as np
import torch

from oww_distill.model import TinyEmbeddingStudent
from oww_distill.tflite_export import (
    StreamingTFLiteRuntime,
    load_student_checkpoint,
    normalize_features,
    quantized_features,
)
from scripts.live_hey_pico import TFLiteRealtimeStudentRuntime


ROOT = Path(__file__).resolve().parents[1]


def test_int8_streaming_student_matches_pytorch() -> None:
    checkpoint = load_student_checkpoint(ROOT / "artifacts/fuurin_embed.pt")
    model = TinyEmbeddingStudent()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    raw = np.zeros((1, 197, 32), dtype=np.float32)
    features = normalize_features(raw, checkpoint)
    with torch.inference_mode():
        expected = model(torch.from_numpy(features))[0].numpy()
    runtime = StreamingTFLiteRuntime(ROOT / "artifacts/fuurin_embed.tflite")
    actual = runtime.selected_embeddings(quantized_features(raw, checkpoint)[0])
    expected_centered = expected - expected.mean(axis=-1, keepdims=True)
    actual_centered = actual - actual.mean(axis=-1, keepdims=True)
    cosine = np.sum(expected_centered * actual_centered, axis=-1) / (
        np.linalg.norm(expected_centered, axis=-1)
        * np.linalg.norm(actual_centered, axis=-1)
        + 1.0e-8
    )
    assert float(cosine.mean()) > 0.95
    assert all(detail["dtype"] == np.int8 for detail in runtime.inputs.values())
    assert all(detail["dtype"] == np.int8 for detail in runtime.outputs.values())
    for index in range(6):
        assert (
            runtime.inputs[f"state_{index}"]["quantization"]
            == runtime.outputs[f"next_state_{index}"]["quantization"]
        )


def test_realtime_tflite_update_matches_continuous_frame_sequence() -> None:
    rng = np.random.default_rng(17)
    pcm = rng.integers(-1000, 1001, size=33_280, dtype=np.int16)
    live = TFLiteRealtimeStudentRuntime(
        ROOT / "artifacts/fuurin_embed.tflite",
        ROOT / "artifacts/fuurin_embed.pt",
        ROOT / "artifacts/fuurin_frontend.npz",
    )
    live.start_stream(pcm[:32_000])
    actual = live.update_stream(pcm[32_000:])

    features = live._features(pcm)
    reference_runtime = StreamingTFLiteRuntime(ROOT / "artifacts/fuurin_embed.tflite")
    frames = reference_runtime.frame_embeddings(features)
    expected = frames[[83 + 8 * index for index in range(16)]]
    np.testing.assert_allclose(actual, expected, atol=1.0e-6)
