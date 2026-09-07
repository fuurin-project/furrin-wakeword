#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from oww_distill.classifier import TorchConvAttentionWakeClassifier
from oww_distill.tflite_head_export import (
    TFLiteWakeHeadRuntime,
    build_conv_attention_head,
    convert_head,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    positive = labels == 1
    negative = ~positive
    accepted = scores >= threshold
    return {
        "recall": float(accepted[positive].mean()),
        "false_accept_rate": float(accepted[negative].mean()),
        "false_accept_count": int(accepted[negative].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the selected wake head as full INT8 TFLite")
    parser.add_argument("--head", type=Path, default=Path("artifacts/hey_pico_head.pt"))
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("data/cache/hey_pico_piper_open_speech_student_generic_v1_delta_streaming.npz"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/hey_pico_head.tflite"))
    parser.add_argument("--metrics", type=Path, default=Path("runs/tflite_head/metrics.json"))
    parser.add_argument("--calibration-clips", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    checkpoint = torch.load(args.head, map_location="cpu", weights_only=True)
    with np.load(args.cache, allow_pickle=False) as cache:
        embeddings = cache["embeddings"].astype(np.float32)
        labels = cache["labels"].astype(np.uint8)
        splits = cache["splits"].astype(np.uint8)
    rng = np.random.default_rng(args.seed)
    candidates = np.flatnonzero(splits == 0)
    calibration = embeddings[
        rng.choice(candidates, min(args.calibration_clips, len(candidates)), replace=False)
    ]
    test = splits == 2

    module, concrete = build_conv_attention_head(checkpoint)
    convert_head(concrete, module, calibration, args.output)
    runtime = TFLiteWakeHeadRuntime(args.output)
    detector = TorchConvAttentionWakeClassifier(args.head)
    float_scores = np.array([detector.score(value) for value in embeddings[test]], np.float32)
    int8_scores = runtime.scores(embeddings[test])
    threshold = detector.threshold
    report = {
        "head": str(args.head),
        "model": str(args.output),
        "bytes": args.output.stat().st_size,
        "sha256": sha256(args.output),
        "calibration_clips": int(len(calibration)),
        "test_clips": int(test.sum()),
        "full_integer_io": runtime.input["dtype"] == np.int8 and runtime.output["dtype"] == np.int8,
        "threshold": threshold,
        "score_mean_abs": float(np.mean(np.abs(int8_scores - float_scores))),
        "score_max_abs": float(np.max(np.abs(int8_scores - float_scores))),
        "decision_flips": int(np.count_nonzero((int8_scores >= threshold) != (float_scores >= threshold))),
        "float32": metrics(labels[test], float_scores, threshold),
        "int8": metrics(labels[test], int8_scores, threshold),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
