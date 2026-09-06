#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from sklearn.neural_network import MLPClassifier

from oww_distill.classifier import BinaryWakeClassifier, DistilledStudentRuntime


SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 32_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Hey Pico head for the 34k backbone")
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
        "--output", type=Path, default=Path("artifacts/hey_pico_34k_head.npz")
    )
    parser.add_argument(
        "--metrics", type=Path, default=Path("runs/hey_pico_34k_head/metrics.json")
    )
    parser.add_argument("--time-shifts", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--minimum-recall", type=float, default=0.90)
    parser.add_argument("--regularization", type=float, default=0.1)
    parser.add_argument(
        "--partial-negative-weight", type=float, default=8.0,
        help="Extra training weight for synthetic Hey-only and Pico-only clips",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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
            # UNKNOWN_TEXTS repeats every 20 sources. Splitting its eight
            # complete cycles 6/1/1 guarantees every phrase, especially the
            # Hey-only and Pico-only negatives, appears in every split.
            cycle = source_index // 20
            return 0 if cycle < 6 else 1 if cycle == 6 else 2
    value = int.from_bytes(hashlib.sha256(group_name.encode()).digest()[:8], "little") % 100
    return 0 if value < 80 else 1 if value < 90 else 2


def hard_negative_kind(path: Path) -> str:
    # The retained Edge-TTS generator assigned UNKNOWN_TEXTS by source index.
    # UNKNOWN_TEXTS[8] is "Pico" and UNKNOWN_TEXTS[9] is "Hey".
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


def extract(
    rows: list[tuple[Path, int, str]], runtime: DistilledStudentRuntime,
    time_shifts: int, batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    embeddings: list[np.ndarray] = []
    labels: list[int] = []
    splits: list[int] = []
    kinds: list[str] = []
    pending: list[np.ndarray] = []
    pending_meta: list[tuple[int, int, str]] = []

    def flush() -> None:
        if not pending:
            return
        frames = runtime.embed_feature_batch(np.stack(pending))
        embeddings.append(frames.reshape(len(frames), -1))
        labels.extend(item[0] for item in pending_meta)
        splits.extend(item[1] for item in pending_meta)
        kinds.extend(item[2] for item in pending_meta)
        pending.clear()
        pending_meta.clear()

    started = time.monotonic()
    for index, (path, label, kind) in enumerate(rows, 1):
        count = time_shifts if kind != "noise" else 1
        path_split = split_for(group(path, kind))
        for clip in shifted(read_pcm(path), count):
            pending.append(runtime.features_pcm(clip))
            pending_meta.append((label, path_split, kind))
            if len(pending) >= batch_size:
                flush()
        if index == 1 or index % 200 == 0 or index == len(rows):
            print(f"extract {index}/{len(rows)}", flush=True)
    flush()
    print(f"extraction_seconds={time.monotonic() - started:.2f}", flush=True)
    return (
        np.concatenate(embeddings), np.asarray(labels, dtype=np.uint8),
        np.asarray(splits, dtype=np.uint8), np.asarray(kinds),
    )


def choose_threshold(labels: np.ndarray, scores: np.ndarray, minimum_recall: float) -> float:
    positives = np.sort(scores[labels == 1])
    allowed_misses = int(np.floor((1.0 - minimum_recall) * len(positives)))
    return float(positives[min(allowed_misses, len(positives) - 1)])


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float, kinds: np.ndarray) -> dict:
    predicted = scores >= threshold
    positive = labels == 1
    negative = ~positive
    report = {
        "clips": int(len(labels)),
        "positives": int(positive.sum()),
        "negatives": int(negative.sum()),
        "threshold": threshold,
        "recall": float(np.mean(predicted[positive])),
        "false_accept_rate": float(np.mean(predicted[negative])),
        "positive_score_p10": float(np.percentile(scores[positive], 10)),
        "negative_score_p99": float(np.percentile(scores[negative], 99)),
    }
    report["negative_false_accept_rate_by_kind"] = {
        kind: float(np.mean(predicted[(kinds == kind) & negative]))
        for kind in sorted(set(kinds[negative]))
    }
    return report


def main() -> None:
    args = parse_args()
    if not 0 < args.minimum_recall <= 1 or args.time_shifts < 1:
        raise SystemExit("invalid minimum recall or time shifts")
    positive = wav_files(args.positive_dir)
    hard_negative = wav_files(args.hard_negative_dir)
    noise = wav_files(args.noise_dir)
    rows = (
        [(path, 1, "positive") for path in positive]
        + [(path, 0, hard_negative_kind(path)) for path in hard_negative]
        + [(path, 0, "noise") for path in noise]
    )
    runtime = DistilledStudentRuntime(args.checkpoint, args.constants, args.device)
    embeddings, labels, splits, kinds = extract(
        rows, runtime, args.time_shifts, args.batch_size
    )
    train, validation, test = splits == 0, splits == 1, splits == 2
    if not all(np.any(mask & (labels == value)) for mask in (train, validation, test) for value in (0, 1)):
        raise SystemExit("every split must contain positives and negatives")
    feature_mean = embeddings[train].mean(axis=0)
    feature_scale = np.maximum(embeddings[train].std(axis=0), 1e-5)
    normalized = (embeddings - feature_mean) / feature_scale
    counts = np.bincount(labels[train], minlength=2)
    sample_weight = len(labels[train]) / (2.0 * counts[labels[train]])
    partial = np.isin(kinds[train], ("hey_only", "pico_only"))
    sample_weight[partial] *= args.partial_negative_weight
    classifier = MLPClassifier(
        hidden_layer_sizes=(args.hidden_dim,), activation="relu", solver="adam",
        alpha=args.regularization, batch_size=min(args.batch_size, int(train.sum())),
        learning_rate_init=0.001, max_iter=500, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=30, random_state=args.seed,
    )
    classifier.fit(normalized[train], labels[train], sample_weight=sample_weight)
    validation_scores = classifier.predict_proba(normalized[validation])[:, 1]
    threshold = choose_threshold(labels[validation], validation_scores, args.minimum_recall)
    wake = BinaryWakeClassifier(
        hidden_weights=classifier.coefs_[0], hidden_bias=classifier.intercepts_[0],
        output_weights=classifier.coefs_[1][:, 0], output_bias=float(classifier.intercepts_[1][0]),
        feature_mean=feature_mean, feature_scale=feature_scale,
        threshold=threshold, name="hey_pico",
    )
    wake.save(args.output)
    report = {
        "status": "provisional_synthetic_edge_tts_not_piper_or_real_voice",
        "backbone": str(args.checkpoint),
        "head": str(args.output),
        "source_wav_clips": {
            "positive": len(positive), "hard_negative": len(hard_negative), "noise": len(noise)
        },
        "expanded_split_clips": {
            "train": int(train.sum()), "validation": int(validation.sum()), "test": int(test.sum())
        },
        "time_shifts": args.time_shifts,
        "partial_negative_weight": args.partial_negative_weight,
        "head_parameters": int(sum(value.size for value in classifier.coefs_ + classifier.intercepts_)),
        "validation": metrics(labels[validation], validation_scores, threshold, kinds[validation]),
        "test": metrics(
            labels[test], classifier.predict_proba(normalized[test])[:, 1], threshold, kinds[test]
        ),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
