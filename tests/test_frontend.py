from pathlib import Path

import numpy as np

from oww_distill.frontend import DspFrontend, FrontendConstants, FFT_SIZE, MEL_BINS


def test_shape_and_finite() -> None:
    constants = FrontendConstants(
        np.hanning(FFT_SIZE).astype(np.float32),
        np.ones((FFT_SIZE // 2 + 1, MEL_BINS), dtype=np.float32),
    )
    output = DspFrontend(constants)(np.zeros(32_000, dtype=np.int16))
    assert output.shape == (197, 32)
    assert np.isfinite(output).all()
    assert DspFrontend(constants).streaming_clip(
        np.zeros(32_000, dtype=np.int16)
    ).shape == (197, 32)


def test_constants_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "constants.npz"
    expected = FrontendConstants(
        np.arange(FFT_SIZE, dtype=np.float32),
        np.ones((FFT_SIZE // 2 + 1, MEL_BINS), dtype=np.float32),
    )
    expected.save(path)
    actual = FrontendConstants.load(path)
    np.testing.assert_array_equal(actual.window, expected.window)
    np.testing.assert_array_equal(actual.mel_weights, expected.mel_weights)
