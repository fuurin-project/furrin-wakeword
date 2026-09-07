#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from oww_distill.classifier import TorchConvAttentionWakeClassifier
from oww_distill.frontend import DspFrontend, FrontendConstants
from oww_distill.tflite_export import (
    StreamingTFLiteRuntime,
    load_student_checkpoint,
    quantized_features,
    streaming_aligned_features,
)
from oww_distill.tflite_head_export import TFLiteWakeHeadRuntime


def centered_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean(axis=-1, keepdims=True)
    right = right - right.mean(axis=-1, keepdims=True)
    numerator = np.sum(left * right, axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return float(np.mean(numerator / np.maximum(denominator, 1.0e-8)))


def head_scores(
    detector: TorchConvAttentionWakeClassifier,
    embeddings: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    output = []
    with torch.inference_mode():
        for start in range(0, len(embeddings), batch_size):
            tensor = torch.from_numpy(embeddings[start : start + batch_size].astype(np.float32))
            tensor = tensor.to(detector.device)
            tensor = (tensor - detector.feature_mean) / detector.feature_scale
            output.append(torch.sigmoid(detector.model(tensor)).cpu().numpy())
    return np.concatenate(output)


def classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    kinds: np.ndarray,
) -> dict[str, object]:
    positive = labels == 1
    negative = ~positive
    accepted = scores >= threshold
    result: dict[str, object] = {
        "clips": int(len(labels)),
        "positives": int(positive.sum()),
        "negatives": int(negative.sum()),
        "threshold": threshold,
        "recall": float(accepted[positive].mean()) if positive.any() else None,
        "false_accept_rate": float(accepted[negative].mean()) if negative.any() else None,
        "false_accept_count": int(accepted[negative].sum()),
    }
    result["negative_far_by_kind"] = {
        str(kind): float(accepted[negative & (kinds == kind)].mean())
        for kind in np.unique(kinds[negative])
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate an INT8 streaming student through the selected Hey Pico head"
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/fuurin_embed.pt"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/fuurin_embed.tflite"))
    parser.add_argument("--constants", type=Path, default=Path("artifacts/fuurin_frontend.npz"))
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path(
            "data/cache/hey_pico_piper_open_speech_student_generic_v1_delta_streaming.npz"
        ),
    )
    parser.add_argument("--head", type=Path, default=Path("artifacts/hey_pico_head.pt"))
    parser.add_argument(
        "--tflite-head",
        type=Path,
        help="Also evaluate a full-INT8 TFLite head on the converted embeddings",
    )
    parser.add_argument("--output", type=Path, default=Path("runs/tflite_student/hey_pico.json"))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--threshold", type=float, help="Override the Float32 head threshold")
    parser.add_argument("--minimum-recall", type=float, default=0.90)
    parser.add_argument("--max-clips", type=int, default=0, help="0 evaluates the full test split")
    args = parser.parse_args()
    if args.max_clips < 0:
        raise SystemExit("--max-clips must not be negative")
    if not 0 < args.minimum_recall <= 1:
        raise SystemExit("--minimum-recall must be in (0, 1]")

    checkpoint = load_student_checkpoint(args.checkpoint)
    with np.load(args.embedding_cache, allow_pickle=False) as cache:
        if str(cache["alignment"]) != "streaming":
            raise SystemExit("embedding cache must use streaming alignment")
        split_value = 1 if args.split == "validation" else 2
        test_indices = np.flatnonzero(cache["splits"] == split_value)
        if args.max_clips:
            test_indices = test_indices[: args.max_clips]
        reference_embeddings = cache["embeddings"][test_indices].astype(np.float32)
        labels = cache["labels"][test_indices].astype(np.uint8)
        kinds = cache["kinds"][test_indices].astype(str)
        paths = cache["paths"][test_indices].astype(str)

    runtime = StreamingTFLiteRuntime(args.model)
    frontend = DspFrontend(FrontendConstants.load(args.constants))
    converted = []
    started = time.monotonic()
    for index, (path, label, kind) in enumerate(zip(paths, labels, kinds, strict=True), 1):
        raw_features = streaming_aligned_features(Path(path), int(label), kind, frontend)
        features = quantized_features(raw_features[None], checkpoint)[0]
        converted.append(runtime.selected_embeddings(features))
        if index % 100 == 0 or index == len(paths):
            elapsed = time.monotonic() - started
            print(f"TFLite clips {index}/{len(paths)} ({elapsed:.1f} s)", flush=True)
    converted_embeddings = np.stack(converted)

    detector = TorchConvAttentionWakeClassifier(args.head)
    reference_scores = head_scores(detector, reference_embeddings)
    converted_scores = head_scores(detector, converted_embeddings)
    threshold = detector.threshold if args.threshold is None else args.threshold
    reference_accept = reference_scores >= threshold
    converted_accept = converted_scores >= threshold
    report = {
        "status": "offline_clip_metrics_not_continuous_or_device_evidence",
        "model": str(args.model),
        "head": str(args.head),
        "split": args.split,
        "test_clips": int(len(labels)),
        "embedding": {
            "int8_vs_reference_centered_cosine": centered_cosine(
                converted_embeddings, reference_embeddings
            ),
            "max_abs": float(np.max(np.abs(converted_embeddings - reference_embeddings))),
            "mean_abs": float(np.mean(np.abs(converted_embeddings - reference_embeddings))),
        },
        "score": {
            "mean_abs": float(np.mean(np.abs(converted_scores - reference_scores))),
            "max_abs": float(np.max(np.abs(converted_scores - reference_scores))),
            "decision_flips": int(np.count_nonzero(reference_accept != converted_accept)),
        },
        "reference_student": classification_metrics(
            labels, reference_scores, threshold, kinds
        ),
        "int8_tflite_student": classification_metrics(
            labels, converted_scores, threshold, kinds
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    if args.tflite_head is not None:
        int8_head = TFLiteWakeHeadRuntime(args.tflite_head)
        full_int8_scores = int8_head.scores(converted_embeddings)
        report["full_int8_score"] = {
            "head": str(args.tflite_head),
            "mean_abs_vs_float_reference": float(
                np.mean(np.abs(full_int8_scores - reference_scores))
            ),
            "max_abs_vs_float_reference": float(
                np.max(np.abs(full_int8_scores - reference_scores))
            ),
            "decision_flips_vs_float_reference": int(
                np.count_nonzero((full_int8_scores >= threshold) != reference_accept)
            ),
        }
        report["full_int8_tflite_student_and_head"] = classification_metrics(
            labels, full_int8_scores, threshold, kinds
        )
    if args.split == "validation":
        positive_scores = np.sort(converted_scores[labels == 1])
        allowed_misses = int(np.floor((1.0 - args.minimum_recall) * len(positive_scores)))
        calibrated_threshold = float(
            positive_scores[min(allowed_misses, len(positive_scores) - 1)]
        )
        report["int8_calibrated_threshold"] = calibrated_threshold
        report["int8_at_calibrated_threshold"] = classification_metrics(
            labels, converted_scores, calibrated_threshold, kinds
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
