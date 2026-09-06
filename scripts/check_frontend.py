#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter

from oww_distill.frontend import DspFrontend, FrontendConstants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--constants", type=Path, default=Path("artifacts/fuurin_frontend.npz"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    cases = {
        "silence": np.zeros(32_000, dtype=np.int16),
        "impulse": np.pad(np.array([20_000], dtype=np.int16), (16_000, 15_999)),
        "noise": rng.integers(-20_000, 20_001, 32_000, dtype=np.int16),
        "tone_440": (12_000 * np.sin(2 * np.pi * 440 * np.arange(32_000) / 16_000)).astype(np.int16),
    }
    interpreter = Interpreter(model_path=str(args.model))
    interpreter.resize_tensor_input(0, [1, 32_000], strict=True)
    interpreter.allocate_tensors()
    dsp = DspFrontend(FrontendConstants.load(args.constants))
    report = {}
    for name, pcm in cases.items():
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], pcm[None].astype(np.float32))
        interpreter.invoke()
        ref = np.squeeze(interpreter.get_tensor(interpreter.get_output_details()[0]["index"])) / 10 + 2
        actual = dsp(pcm)
        error = np.abs(ref - actual)
        report[name] = {
            "shape": list(actual.shape),
            "max_abs_error": float(error.max()),
            "mean_abs_error": float(error.mean()),
        }
    print(json.dumps(report, indent=2))
    if any(row["max_abs_error"] > 5e-5 for row in report.values()):
        raise SystemExit("frontend parity failed")


if __name__ == "__main__":
    main()
