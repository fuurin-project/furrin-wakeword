from pathlib import Path

import numpy as np

from oww_distill.classifier import TemporalWakeClassifier, load_wake_classifier


def temporal_classifier() -> TemporalWakeClassifier:
    rng = np.random.default_rng(17)
    return TemporalWakeClassifier(
        depthwise_weights=rng.normal(size=(96, 3)),
        depthwise_bias=rng.normal(size=96),
        pointwise_weights=rng.normal(size=(8, 96)),
        pointwise_bias=rng.normal(size=8),
        output_weights=rng.normal(size=16),
        output_bias=0.25,
        feature_mean=rng.normal(size=96),
        feature_scale=rng.uniform(0.1, 2.0, size=96),
        threshold=0.4,
    )


def test_temporal_classifier_round_trip(tmp_path: Path) -> None:
    expected = temporal_classifier()
    frames = np.random.default_rng(18).normal(size=(16, 96)).astype(np.float32)
    path = tmp_path / "head.npz"
    expected.save(path)
    actual = load_wake_classifier(path)
    assert isinstance(actual, TemporalWakeClassifier)
    np.testing.assert_allclose(actual.score(frames), expected.score(frames), rtol=0, atol=1e-7)


def test_temporal_classifier_rejects_wrong_shape() -> None:
    classifier = temporal_classifier()
    try:
        classifier.score(np.zeros((16, 95), dtype=np.float32))
    except ValueError as exc:
        assert "must have shape" in str(exc)
    else:
        raise AssertionError("wrong feature dimension was accepted")
