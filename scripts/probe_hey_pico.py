#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.neural_network import MLPClassifier

from oww_distill.classifier import DistilledStudentRuntime


SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 32_000
TEACHER_WINDOW_FRAMES = 76
TEACHER_STEP_FRAMES = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare openWakeWord teacher and distilled-student embeddings with "
            "an otherwise identical Hey Pico head and dataset split."
        )
    )
    parser.add_argument(
        "--positive-dir", type=Path,
        default=Path("/home/user/Workspace/wakeup_word/data/raw/hey_pico"),
    )
    parser.add_argument(
        "--hard-negative-dir", type=Path,
        default=Path("/home/user/Workspace/wakeup_word/data/raw/unknown"),
    )
    parser.add_argument(
        "--noise-dir", type=Path,
        default=Path("/home/user/Workspace/wakeup_word/data/raw/noise"),
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("runs/student_balanced_57078/student_best.pt"),
    )
    parser.add_argument(
        "--constants", type=Path, default=Path("artifacts/fuurin_frontend.npz")
    )
    parser.add_argument(
        "--cache", type=Path,
        default=Path("data/cache/hey_pico_matched_edge_tts.npz"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/hey_pico_teacher_student_probe/metrics.json"),
    )
    parser.add_argument("--time-shifts", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--minimum-recall", type=float, default=0.90)
    parser.add_argument("--regularization", type=float, default=0.1)
    parser.add_argument("--partial-negative-weight", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def wav_files(directory: Path) -> list[Path]:
    paths = sorted(directory.rglob("*.wav"))
    if not paths:
        raise SystemExit(f"No WAV files found in {directory}")
    return paths


def group(path: Path, kind: str) -> str:
    if "_nohash_" in path.stem:
        return path.stem.split("_nohash_")[0]
    return f"{kind}:{path.stem}"


def split_for(group_name: str) -> int:
    if group_name.startswith("speaker_"):
        try:
            source_index = int(group_name.rsplit("_", 1)[1])
        except ValueError:
            pass
        else:
            cycle = source_index // 20
            return 0 if cycle < 6 else 1 if cycle == 6 else 2
    value = int.from_bytes(hashlib.sha256(group_name.encode()).digest()[:8], "little") % 100
    return 0 if value < 80 else 1 if value < 90 else 2


def hard_negative_kind(path: Path) -> str:
    prefix = path.stem.split("_nohash_")[0]
    try:
        source_index = int(prefix.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return "hard_negative"
    phrase_index = source_index % 20
    if phrase_index == 8:
        return "pico_only"
    if phrase_index == 9:
        return "hey_only"
    return "hard_negative"


def read_pcm(path: Path) -> np.ndarray:
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="int16", always_2d=False)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"{path}: expected 16 kHz, got {sample_rate}")
    if audio.ndim == 2:
        audio = np.rint(audio.astype(np.float32).mean(axis=1)).astype(np.int16)
    return np.asarray(audio, dtype=np.int16)


def shifted(pcm: np.ndarray, count: int) -> list[np.ndarray]:
    pcm = np.asarray(pcm, dtype=np.int16).reshape(-1)
    if len(pcm) >= WINDOW_SAMPLES:
        return [DistilledStudentRuntime.center_pad_or_trim(pcm)]
    starts = np.linspace(0, WINDOW_SAMPLES - len(pcm), num=count, dtype=np.int32)
    clips = []
    for start in starts:
        clip = np.zeros(WINDOW_SAMPLES, dtype=np.int16)
        clip[start : start + len(pcm)] = pcm
        clips.append(clip)
    return clips


def dataset_rows(args: argparse.Namespace) -> list[tuple[Path, int, str]]:
    return (
        [(path, 1, "positive") for path in wav_files(args.positive_dir)]
        + [
            (path, 0, hard_negative_kind(path))
            for path in wav_files(args.hard_negative_dir)
        ]
        + [(path, 0, "noise") for path in wav_files(args.noise_dir)]
    )


def teacher_embed_features(features: np.ndarray, teacher, batch_size: int) -> np.ndarray:
    starts = range(
        0,
        features.shape[1] - TEACHER_WINDOW_FRAMES + 1,
        TEACHER_STEP_FRAMES,
    )
    windows = np.concatenate(
        [features[:, start : start + TEACHER_WINDOW_FRAMES, :, None] for start in starts],
        axis=0,
    )
    expected_frames = len(tuple(starts))
    outputs = []
    for start in range(0, len(windows), batch_size):
        prediction = teacher.embedding_model_predict(
            windows[start : start + batch_size].astype(np.float32)
        )
        outputs.append(np.asarray(prediction, dtype=np.float32).reshape(-1, 96))
    # Windows were concatenated by time index, so restore clip-major ordering.
    stacked = np.concatenate(outputs).reshape(expected_frames, len(features), 96)
    return stacked.transpose(1, 0, 2)


def build_cache(args: argparse.Namespace) -> dict[str, np.ndarray]:
    os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
    from openwakeword.utils import AudioFeatures

    rows = dataset_rows(args)
    runtime = DistilledStudentRuntime(args.checkpoint, args.constants, args.device)
    teacher = AudioFeatures(inference_framework="tflite", ncpu=4)
    student_parts: list[np.ndarray] = []
    teacher_parts: list[np.ndarray] = []
    labels: list[int] = []
    splits: list[int] = []
    kinds: list[str] = []
    paths: list[str] = []
    shift_indices: list[int] = []
    pending: list[np.ndarray] = []
    pending_meta: list[tuple[int, int, str, str, int]] = []

    def flush() -> None:
        if not pending:
            return
        feature_batch = np.stack(pending)
        student_parts.append(runtime.embed_feature_batch(feature_batch).astype(np.float16))
        teacher_parts.append(
            teacher_embed_features(feature_batch, teacher, args.batch_size).astype(np.float16)
        )
        for label, split, kind, path, shift_index in pending_meta:
            labels.append(label)
            splits.append(split)
            kinds.append(kind)
            paths.append(path)
            shift_indices.append(shift_index)
        pending.clear()
        pending_meta.clear()

    started = time.monotonic()
    for index, (path, label, kind) in enumerate(rows, 1):
        count = args.time_shifts if kind != "noise" else 1
        split = split_for(group(path, kind))
        for shift_index, clip in enumerate(shifted(read_pcm(path), count)):
            pending.append(runtime.features_pcm(clip))
            pending_meta.append((label, split, kind, str(path), shift_index))
            if len(pending) >= args.batch_size:
                flush()
        if index == 1 or index % 100 == 0 or index == len(rows):
            elapsed = max(time.monotonic() - started, 1e-6)
            print(f"extract {index}/{len(rows)} ({index / elapsed:.2f} source clips/s)", flush=True)
    flush()
    values = {
        "format_version": np.array(1, dtype=np.int32),
        "student": np.concatenate(student_parts),
        "teacher": np.concatenate(teacher_parts),
        "labels": np.asarray(labels, dtype=np.uint8),
        "splits": np.asarray(splits, dtype=np.uint8),
        "kinds": np.asarray(kinds),
        "paths": np.asarray(paths),
        "shift_indices": np.asarray(shift_indices, dtype=np.uint8),
    }
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.cache, **values)
    print(f"saved {args.cache} with {len(labels)} expanded clips", flush=True)
    return values


def load_cache(args: argparse.Namespace) -> dict[str, np.ndarray]:
    if args.rebuild_cache or not args.cache.exists():
        return build_cache(args)
    with np.load(args.cache, allow_pickle=False) as cache:
        if int(cache["format_version"]) != 1:
            raise SystemExit("unsupported probe cache format")
        return {name: cache[name] for name in cache.files}


def choose_threshold(labels: np.ndarray, scores: np.ndarray, minimum_recall: float) -> float:
    positives = np.sort(scores[labels == 1])
    allowed_misses = int(np.floor((1.0 - minimum_recall) * len(positives)))
    return float(positives[min(allowed_misses, len(positives) - 1)])


def split_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float, kinds: np.ndarray
) -> dict[str, object]:
    predicted = scores >= threshold
    positive = labels == 1
    negative = ~positive
    return {
        "clips": int(len(labels)),
        "positives": int(positive.sum()),
        "negatives": int(negative.sum()),
        "threshold": threshold,
        "recall": float(np.mean(predicted[positive])),
        "false_accept_rate": float(np.mean(predicted[negative])),
        "positive_score_p10": float(np.percentile(scores[positive], 10)),
        "negative_score_p99": float(np.percentile(scores[negative], 99)),
        "negative_false_accept_rate_by_kind": {
            kind: float(np.mean(predicted[(kinds == kind) & negative]))
            for kind in sorted(set(kinds[negative]))
        },
    }


def evaluate_backbone(
    name: str,
    embeddings: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    kinds: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    train, validation, test = splits == 0, splits == 1, splits == 2
    flat = embeddings.astype(np.float32).reshape(len(embeddings), -1)
    feature_mean = flat[train].mean(axis=0)
    feature_scale = np.maximum(flat[train].std(axis=0), 1e-5)
    normalized = (flat - feature_mean) / feature_scale
    counts = np.bincount(labels[train], minlength=2)
    sample_weight = len(labels[train]) / (2.0 * counts[labels[train]])
    sample_weight[np.isin(kinds[train], ("hey_only", "pico_only"))] *= (
        args.partial_negative_weight
    )
    classifier = MLPClassifier(
        hidden_layer_sizes=(args.hidden_dim,), activation="relu", solver="adam",
        alpha=args.regularization, batch_size=min(args.batch_size, int(train.sum())),
        learning_rate_init=0.001, max_iter=500, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=30, random_state=args.seed,
    )
    classifier.fit(normalized[train], labels[train], sample_weight=sample_weight)
    validation_scores = classifier.predict_proba(normalized[validation])[:, 1]
    threshold = choose_threshold(labels[validation], validation_scores, args.minimum_recall)
    return {
        "name": name,
        "embedding_shape": list(embeddings.shape[1:]),
        "head": "flatten_16x96_mlp_32",
        "head_parameters": int(sum(value.size for value in classifier.coefs_ + classifier.intercepts_)),
        "iterations": int(classifier.n_iter_),
        "validation": split_metrics(
            labels[validation], validation_scores, threshold, kinds[validation]
        ),
        "test": split_metrics(
            labels[test], classifier.predict_proba(normalized[test])[:, 1], threshold,
            kinds[test],
        ),
    }


def main() -> None:
    args = parse_args()
    if not 0 < args.minimum_recall <= 1 or args.time_shifts < 1:
        raise SystemExit("invalid minimum recall or time shifts")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    values = load_cache(args)
    labels = values["labels"].astype(np.uint8)
    splits = values["splits"].astype(np.uint8)
    kinds = values["kinds"].astype(str)
    if set(np.unique(splits)) != {0, 1, 2}:
        raise SystemExit("probe cache must contain train, validation, and test splits")
    dataset_fingerprint = hashlib.sha256(
        "\n".join(
            f"{path}\t{shift}\t{split}\t{label}\t{kind}"
            for path, shift, split, label, kind in zip(
                values["paths"].astype(str), values["shift_indices"], splits, labels,
                kinds, strict=True,
            )
        ).encode()
    ).hexdigest()
    results = [
        evaluate_backbone(
            "teacher", values["teacher"], labels, splits, kinds, args
        ),
        evaluate_backbone(
            "distilled_student", values["student"], labels, splits, kinds, args
        ),
    ]
    report = {
        "status": "matched_probe_synthetic_edge_tts_not_deployment_evidence",
        "cache": str(args.cache),
        "checkpoint": str(args.checkpoint),
        "dataset_fingerprint": dataset_fingerprint,
        "split_clips": {
            "train": int(np.sum(splits == 0)),
            "validation": int(np.sum(splits == 1)),
            "test": int(np.sum(splits == 2)),
        },
        "probe": {
            "same_dataset_split_head_seed_and_update_budget": True,
            "time_shifts": args.time_shifts,
            "hidden_dim": args.hidden_dim,
            "regularization": args.regularization,
            "partial_negative_weight": args.partial_negative_weight,
            "minimum_validation_recall": args.minimum_recall,
            "seed": args.seed,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
