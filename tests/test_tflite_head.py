from pathlib import Path

import numpy as np

from oww_distill.classifier import TorchConvAttentionWakeClassifier
from oww_distill.tflite_head_export import TFLiteWakeHeadRuntime


ROOT = Path(__file__).resolve().parents[1]


def test_int8_head_matches_float_head() -> None:
    rng = np.random.default_rng(17)
    embeddings = rng.normal(size=(16, 96)).astype(np.float32)
    float_head = TorchConvAttentionWakeClassifier(ROOT / "artifacts/hey_pico_head.pt")
    int8_head = TFLiteWakeHeadRuntime(ROOT / "artifacts/hey_pico_head.tflite")
    float_score = float_head.score(embeddings)
    int8_score = int8_head.score(embeddings)
    assert abs(float_score - int8_score) < 0.08
    assert int8_head.input["dtype"] == np.int8
    assert int8_head.output["dtype"] == np.int8
