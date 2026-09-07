#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import torch

from oww_distill.model import TinyEmbeddingStudent
from oww_distill.frontend import DspFrontend, FrontendConstants
from oww_distill.tflite_export import (
    StreamingTFLiteRuntime,
    build_streaming_student,
    convert_streaming_student,
    load_student_checkpoint,
    normalize_features,
    quantized_features,
    representative_steps,
    streaming_aligned_features,
    zero_states,
)


def centered_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean(axis=-1, keepdims=True)
    right = right - right.mean(axis=-1, keepdims=True)
    numerator = np.sum(left * right, axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return float(np.mean(numerator / np.maximum(denominator, 1.0e-8)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the distilled student as stateful Float32 and full-INT8 TFLite"
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/fuurin_embed.pt"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache/distill_smoke_4k.npz"))
    parser.add_argument(
        "--metadata-cache",
        type=Path,
        help="Optional wake cache whose paths are rebuilt with streaming alignment",
    )
    parser.add_argument("--constants", type=Path, default=Path("artifacts/fuurin_frontend.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tflite_student"))
    parser.add_argument("--calibration-clips", type=int, default=32)
    parser.add_argument("--validation-clips", type=int, default=16)
    parser.add_argument("--calibration-stride", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if min(args.calibration_clips, args.validation_clips, args.calibration_stride) < 1:
        raise SystemExit("clip counts and calibration stride must be positive")

    checkpoint = load_student_checkpoint(args.checkpoint)
    if args.metadata_cache is not None:
        with np.load(args.metadata_cache, allow_pickle=False) as cache:
            paths = cache["paths"].astype(str)
            labels = cache["labels"].astype(np.uint8)
            kinds = cache["kinds"].astype(str)
        rng = np.random.default_rng(args.seed)
        calibration_parts = []
        validation_parts = []
        unique_kinds = np.unique(kinds)
        per_kind = max(1, int(np.ceil(args.calibration_clips / len(unique_kinds))))
        frontend = DspFrontend(FrontendConstants.load(args.constants))
        selected = []
        for kind in unique_kinds:
            candidates = np.flatnonzero(kinds == kind)
            selected.extend(
                rng.choice(candidates, min(per_kind, len(candidates)), replace=False).tolist()
            )
        selected = selected[: args.calibration_clips]
        remaining = np.setdiff1d(np.arange(len(paths)), selected)
        validation_indices = rng.choice(
            remaining, min(args.validation_clips, len(remaining)), replace=False
        )
        for index in selected:
            calibration_parts.append(
                streaming_aligned_features(
                    Path(paths[index]), int(labels[index]), kinds[index], frontend
                )
            )
        for index in validation_indices:
            validation_parts.append(
                streaming_aligned_features(
                    Path(paths[index]), int(labels[index]), kinds[index], frontend
                )
            )
        calibration = quantized_features(np.stack(calibration_parts), checkpoint)
        validation = quantized_features(np.stack(validation_parts), checkpoint)
    else:
        with np.load(args.cache, allow_pickle=False) as cache:
            raw_features = cache["features"]
            rng = np.random.default_rng(args.seed)
            if "splits" in cache.files:
                candidates = np.flatnonzero(cache["splits"] == 0)
                validation_candidates = np.flatnonzero(cache["splits"] != 0)
                calibration_indices = rng.choice(
                    candidates, min(args.calibration_clips, len(candidates)), replace=False
                )
                validation_indices = rng.choice(
                    validation_candidates,
                    min(args.validation_clips, len(validation_candidates)),
                    replace=False,
                )
            else:
                order = rng.permutation(len(raw_features))
                calibration_count = min(args.calibration_clips, len(order))
                validation_count = min(args.validation_clips, len(order) - calibration_count)
                calibration_indices = order[:calibration_count]
                validation_indices = order[
                    calibration_count : calibration_count + validation_count
                ]
            calibration = quantized_features(raw_features[calibration_indices], checkpoint)
            validation = quantized_features(raw_features[validation_indices], checkpoint)

    module, concrete = build_streaming_student(checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    float_path = args.output_dir / "fuurin_embed_stream_float32.tflite"
    int8_path = args.output_dir / "fuurin_embed_stream_int8.tflite"
    convert_streaming_student(concrete, module, float_path)

    def representative_dataset():
        yield from representative_steps(calibration, module, args.calibration_stride)

    convert_streaming_student(
        concrete, module, int8_path, representative_dataset=representative_dataset
    )

    architecture = checkpoint["architecture"]
    torch_model = TinyEmbeddingStudent(
        feature_bins=int(architecture["feature_bins"]),
        channels=int(architecture["channels"]),
    )
    torch_model.load_state_dict(checkpoint["model_state"])
    torch_model.eval()
    float_runtime = StreamingTFLiteRuntime(float_path)
    int8_runtime = StreamingTFLiteRuntime(int8_path)
    float_values = []
    int8_values = []
    torch_values = []
    with torch.inference_mode():
        for clip in validation:
            normalized = normalize_features(clip[None], checkpoint)
            expected = torch_model(torch.from_numpy(normalized))[0].numpy()
            torch_values.append(expected)
            float_values.append(float_runtime.selected_embeddings(clip))
            int8_values.append(int8_runtime.selected_embeddings(clip))
    torch_values = np.stack(torch_values)
    float_values = np.stack(float_values)
    int8_values = np.stack(int8_values)
    report = {
        "checkpoint": str(args.checkpoint),
        "calibration_source": str(args.metadata_cache or args.cache),
        "streaming_step": {
            "feature_frames": 1,
            "feature_bins": 32,
            "calls_per_80_ms": 8,
            "state_bytes_int8": int(sum(state.size for state in zero_states())),
        },
        "calibration_clips": int(len(calibration)),
        "validation_clips": int(len(validation)),
        "float32": {
            "path": str(float_path),
            "bytes": float_path.stat().st_size,
            "sha256": sha256(float_path),
            "vs_pytorch_centered_cosine": centered_cosine(float_values, torch_values),
            "vs_pytorch_max_abs": float(np.max(np.abs(float_values - torch_values))),
        },
        "int8": {
            "path": str(int8_path),
            "bytes": int8_path.stat().st_size,
            "sha256": sha256(int8_path),
            "vs_pytorch_centered_cosine": centered_cosine(int8_values, torch_values),
            "vs_pytorch_max_abs": float(np.max(np.abs(int8_values - torch_values))),
            "vs_float32_centered_cosine": centered_cosine(int8_values, float_values),
            "full_integer_io": all(
                detail["dtype"] == np.int8
                for detail in (*int8_runtime.inputs.values(), *int8_runtime.outputs.values())
            ),
            "state_input_output_scales_match": all(
                int8_runtime.inputs[f"state_{index}"]["quantization"]
                == int8_runtime.outputs[f"next_state_{index}"]["quantization"]
                for index in range(6)
            ),
            "required_tflite_micro_ops": [
                "ADD",
                "CONCATENATION",
                "CONV_2D",
                "DEPTHWISE_CONV_2D",
                "RESHAPE",
                "STRIDED_SLICE",
            ],
        },
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
