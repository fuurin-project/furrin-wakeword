#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn

from oww_distill.classifier import DistilledStudentRuntime, TemporalWakeClassifier
from oww_distill.wake_phrase import is_partial_kind


SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 32_000
SPLIT_INDEX = {"train": 0, "validation": 1, "test": 2}


@dataclass(frozen=True)
class Row:
    path: Path
    label: int
    kind: str
    split: int
    group: str
    source: str


class TemporalHead(nn.Module):
    def __init__(self, input_dim: int = 96, channels: int = 16, kernel_size: int = 3) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            input_dim, input_dim, kernel_size, padding=kernel_size // 2,
            groups=input_dim,
        )
        self.pointwise = nn.Conv1d(input_dim, channels, 1)
        self.output = nn.Linear(2 * channels, 1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.pointwise(self.depthwise(frames.transpose(1, 2))))
        pooled = torch.cat((hidden.amax(dim=2), hidden.mean(dim=2)), dim=1)
        return self.output(pooled).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an MCU-sized temporal Hey Pico head on Piper and open speech."
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/hey_pico_piper_v1/manifest.jsonl")
    )
    parser.add_argument("--wake-name", default="hey_pico")
    parser.add_argument(
        "--alignment", choices=("random", "streaming"), default="random",
        help="streaming places all speech near the trailing edge with deterministic jitter",
    )
    parser.add_argument(
        "--external-root", type=Path,
        default=Path("/home/user/Workspace/wakeup_word/data/external/extracted"),
    )
    parser.add_argument(
        "--edge-hard-negative-dir", type=Path,
        default=Path("/home/user/Workspace/wakeup_word/data/raw/unknown"),
    )
    parser.add_argument(
        "--edge-noise-dir", type=Path,
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
        default=Path("data/cache/hey_pico_piper_open_speech_student.npz"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/hey_pico_temporal_piper_v1_head.npz"),
    )
    parser.add_argument(
        "--metrics", type=Path,
        default=Path("runs/hey_pico_temporal_piper_v1/metrics.json"),
    )
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--partial-negative-weight", type=float, default=4.0)
    parser.add_argument("--minimum-recall", type=float, default=0.90)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def hash_split(group: str) -> int:
    value = int.from_bytes(hashlib.sha256(group.encode()).digest()[:8], "little") % 100
    return 0 if value < 80 else 1 if value < 90 else 2


def piper_rows(manifest: Path) -> list[Row]:
    base = manifest.parent
    rows = []
    for line in manifest.read_text().splitlines():
        value = json.loads(line)
        rows.append(
            Row(
                path=base / value["path"], label=int(value["label"]),
                kind=str(value["kind"]), split=SPLIT_INDEX[str(value["split"])],
                group=str(value["group"]), source="piper",
            )
        )
    return rows


def external_rows(root: Path) -> list[Row]:
    rows: list[Row] = []
    mini = root / "mini_speech_commands"
    for path in sorted(mini.rglob("*.wav")):
        if path.name.startswith("._"):
            continue
        group = f"speech_commands:{path.stem.split('_nohash_')[0]}"
        rows.append(Row(path, 0, "open_speech", hash_split(group), group, "speech_commands"))
    libri = root / "LibriSpeech" / "dev-clean"
    for path in sorted(libri.rglob("*.flac")):
        speaker = path.relative_to(libri).parts[0]
        group = f"librispeech:{speaker}"
        rows.append(Row(path, 0, "open_speech", hash_split(group), group, "librispeech"))
    fsdd = root / "free-spoken-digit-dataset-1.0.10" / "recordings"
    for path in sorted(fsdd.glob("*.wav")):
        parts = path.stem.split("_")
        speaker = parts[1] if len(parts) > 2 else path.stem
        group = f"fsdd:{speaker}"
        rows.append(Row(path, 0, "open_speech", hash_split(group), group, "fsdd"))
    return rows


def edge_rows(hard_dir: Path, noise_dir: Path) -> list[Row]:
    rows = []
    for path in sorted(hard_dir.rglob("*.wav")):
        prefix = path.stem.split("_nohash_")[0]
        try:
            source_index = int(prefix.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            kind = "edge_hard_negative"
        else:
            phrase_index = source_index % 20
            kind = "pico_only" if phrase_index == 8 else "hey_only" if phrase_index == 9 else "edge_hard_negative"
        group = f"edge:{prefix}"
        rows.append(Row(path, 0, kind, hash_split(group), group, "edge_tts"))
    for path in sorted(noise_dir.rglob("*.wav")):
        group = f"noise:{path.stem.split('_nohash_')[0]}"
        rows.append(Row(path, 0, "noise", hash_split(group), group, "synthetic_noise"))
    return rows


def deterministic_window(path: Path, kind: str, alignment: str) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="int16", always_2d=False)
    if sample_rate != SAMPLE_RATE:
        positions = np.linspace(0, len(audio) - 1, max(1, int(round(len(audio) * SAMPLE_RATE / sample_rate))))
        audio = np.interp(positions, np.arange(len(audio)), audio).astype(np.int16)
    if audio.ndim == 2:
        audio = np.rint(audio.astype(np.float32).mean(axis=1)).astype(np.int16)
    audio = np.asarray(audio, dtype=np.int16).reshape(-1)
    seed = int.from_bytes(hashlib.sha256(str(path).encode()).digest()[:8], "little")
    if alignment == "streaming" and kind != "noise":
        output = np.zeros(WINDOW_SAMPLES, dtype=np.int16)
        jitter = seed % 3201
        end = WINDOW_SAMPLES - jitter
        start = max(0, end - len(audio))
        source_start = max(0, len(audio) - (end - start))
        output[start:end] = audio[source_start : source_start + end - start]
        return output
    if len(audio) >= WINDOW_SAMPLES:
        start = seed % (len(audio) - WINDOW_SAMPLES + 1)
        return audio[start : start + WINDOW_SAMPLES].copy()
    output = np.zeros(WINDOW_SAMPLES, dtype=np.int16)
    start = seed % (WINDOW_SAMPLES - len(audio) + 1)
    output[start : start + len(audio)] = audio
    return output


def build_cache(args: argparse.Namespace) -> dict[str, np.ndarray]:
    rows = piper_rows(args.manifest)
    rows += external_rows(args.external_root)
    rows += edge_rows(args.edge_hard_negative_dir, args.edge_noise_dir)
    runtime = DistilledStudentRuntime(args.checkpoint, args.constants, args.device)
    embeddings: list[np.ndarray] = []
    pending: list[np.ndarray] = []
    started = time.monotonic()

    def flush() -> None:
        if pending:
            embeddings.append(runtime.embed_feature_batch(np.stack(pending)).astype(np.float16))
            pending.clear()

    for index, row in enumerate(rows, 1):
        pcm = deterministic_window(row.path, row.kind, args.alignment)
        pending.append(runtime.frontend.streaming_clip(pcm))
        if len(pending) >= args.batch_size:
            flush()
        if index == 1 or index % 500 == 0 or index == len(rows):
            elapsed = max(time.monotonic() - started, 1e-6)
            print(f"embed {index}/{len(rows)} ({index / elapsed:.1f} clips/s)", flush=True)
    flush()
    values = {
        "format_version": np.array(1, dtype=np.int32),
        "embeddings": np.concatenate(embeddings),
        "labels": np.asarray([row.label for row in rows], dtype=np.uint8),
        "splits": np.asarray([row.split for row in rows], dtype=np.uint8),
        "kinds": np.asarray([row.kind for row in rows]),
        "sources": np.asarray([row.source for row in rows]),
        "groups": np.asarray([row.group for row in rows]),
        "paths": np.asarray([str(row.path) for row in rows]),
        "alignment": np.array(args.alignment),
        "student_checkpoint_sha256": np.array(
            hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        ),
        "manifest_sha256": np.array(hashlib.sha256(args.manifest.read_bytes()).hexdigest()),
    }
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.cache, **values)
    print(f"saved {args.cache}: {len(rows)} clips", flush=True)
    return values


def load_cache(args: argparse.Namespace) -> dict[str, np.ndarray]:
    if args.rebuild_cache or not args.cache.exists():
        return build_cache(args)
    with np.load(args.cache, allow_pickle=False) as cache:
        if int(cache["format_version"]) != 1:
            raise SystemExit("unsupported cache format")
        values = {name: cache[name] for name in cache.files}
    expected_checkpoint = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    expected_manifest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if str(values.get("student_checkpoint_sha256", "")) != expected_checkpoint:
        raise SystemExit("cache uses a different student checkpoint; pass --rebuild-cache")
    if str(values.get("manifest_sha256", "")) != expected_manifest:
        raise SystemExit("cache uses a different dataset manifest; pass --rebuild-cache")
    if str(values.get("alignment", "")) != args.alignment:
        raise SystemExit("cache uses a different alignment; pass --rebuild-cache")
    return values


def choose_threshold(labels: np.ndarray, scores: np.ndarray, minimum_recall: float) -> float:
    positives = np.sort(scores[labels == 1])
    allowed_misses = int(np.floor((1.0 - minimum_recall) * len(positives)))
    return float(positives[min(allowed_misses, len(positives) - 1)])


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float, kinds: np.ndarray, sources: np.ndarray) -> dict:
    predicted = scores >= threshold
    positive, negative = labels == 1, labels == 0
    return {
        "clips": int(len(labels)), "positives": int(positive.sum()), "negatives": int(negative.sum()),
        "threshold": threshold, "recall": float(np.mean(predicted[positive])),
        "false_accept_rate": float(np.mean(predicted[negative])),
        "false_accept_count": int(np.sum(predicted & negative)),
        "positive_score_p10": float(np.percentile(scores[positive], 10)),
        "negative_score_p99": float(np.percentile(scores[negative], 99)),
        "negative_far_by_kind": {
            kind: float(np.mean(predicted[(kinds == kind) & negative]))
            for kind in sorted(set(kinds[negative]))
        },
        "negative_far_by_source": {
            source: float(np.mean(predicted[(sources == source) & negative]))
            for source in sorted(set(sources[negative]))
        },
    }


def model_scores(model: nn.Module, values: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            logits = model(torch.from_numpy(values[start : start + batch_size]).to(device))
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs)


def main() -> None:
    args = parse_args()
    if args.kernel_size % 2 != 1 or args.kernel_size < 1:
        raise SystemExit("kernel size must be positive and odd")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    values = load_cache(args)
    embeddings = values["embeddings"].astype(np.float32)
    labels = values["labels"].astype(np.float32)
    splits = values["splits"].astype(np.uint8)
    kinds = values["kinds"].astype(str)
    sources = values["sources"].astype(str)
    train, validation, test = splits == 0, splits == 1, splits == 2
    if not all(np.any(mask & (labels == value)) for mask in (train, validation, test) for value in (0, 1)):
        raise SystemExit("every split must contain positive and negative clips")
    feature_mean = embeddings[train].mean(axis=(0, 1))
    feature_scale = np.maximum(embeddings[train].std(axis=(0, 1)), 1e-5)
    normalized = (embeddings - feature_mean[None, None, :]) / feature_scale[None, None, :]
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    model = TemporalHead(channels=args.channels, kernel_size=args.kernel_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    counts = np.bincount(labels[train].astype(np.uint8), minlength=2)
    weights = len(labels[train]) / (2.0 * counts[labels[train].astype(np.uint8)])
    partial = np.asarray([is_partial_kind(kind) for kind in kinds[train]])
    weights[partial] *= args.partial_negative_weight
    x_train = torch.from_numpy(normalized[train])
    y_train = torch.from_numpy(labels[train])
    w_train = torch.from_numpy(weights.astype(np.float32))
    generator = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train, w_train),
        batch_size=args.batch_size, shuffle=True, generator=generator,
    )
    best_state = None
    best_far = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y, batch_weight in loader:
            batch_x, batch_y, batch_weight = batch_x.to(device), batch_y.to(device), batch_weight.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(batch_x), batch_y, weight=batch_weight,
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_scores = model_scores(model, normalized[validation], device, args.batch_size)
        threshold = choose_threshold(labels[validation], val_scores, args.minimum_recall)
        val_far = float(np.mean(val_scores[labels[validation] == 0] >= threshold))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_far": val_far, "threshold": threshold})
        if val_far < best_far:
            best_far, best_epoch, stale = val_far, epoch, 0
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch} loss={np.mean(losses):.5f} val_far={val_far:.5f} threshold={threshold:.5f}", flush=True)
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    val_scores = model_scores(model, normalized[validation], device, args.batch_size)
    threshold = choose_threshold(labels[validation], val_scores, args.minimum_recall)
    test_scores = model_scores(model, normalized[test], device, args.batch_size)
    classifier = TemporalWakeClassifier(
        depthwise_weights=model.depthwise.weight.detach().cpu().numpy()[:, 0, :],
        depthwise_bias=model.depthwise.bias.detach().cpu().numpy(),
        pointwise_weights=model.pointwise.weight.detach().cpu().numpy()[:, :, 0],
        pointwise_bias=model.pointwise.bias.detach().cpu().numpy(),
        output_weights=model.output.weight.detach().cpu().numpy()[0],
        output_bias=float(model.output.bias.detach().cpu().numpy()[0]),
        feature_mean=feature_mean, feature_scale=feature_scale,
        threshold=threshold,
        name=args.wake_name,
    )
    classifier.save(args.output)
    # Verify the dependency-free NumPy runtime matches PyTorch.
    check_indices = np.flatnonzero(test)[:32]
    numpy_scores = np.asarray([classifier.score(embeddings[index]) for index in check_indices])
    torch_scores = model_scores(model, normalized[check_indices], device, args.batch_size)
    runtime_max_abs_error = float(np.max(np.abs(numpy_scores - torch_scores)))
    report = {
        "status": "piper_voice_disjoint_plus_open_speech_offline_not_continuous_or_real_voice",
        "wake_name": args.wake_name,
        "backbone": str(args.checkpoint), "head": str(args.output), "cache": str(args.cache),
        "device": str(device),
        "dataset": {
            "clips": int(len(labels)),
            "split_clips": {"train": int(train.sum()), "validation": int(validation.sum()), "test": int(test.sum())},
            "positive_clips": int(labels.sum()),
            "source_counts": {source: int(np.sum(sources == source)) for source in sorted(set(sources))},
            "group_count": int(len(set(values["groups"].astype(str)))),
        },
        "architecture": {
            "type": "depthwise_temporal_conv_pointwise_max_mean",
            "channels": args.channels, "kernel_size": args.kernel_size,
            "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        },
        "training": {"best_epoch": best_epoch, "epochs_ran": len(history), "seconds": time.monotonic() - started, "history": history},
        "validation": metrics(labels[validation], val_scores, threshold, kinds[validation], sources[validation]),
        "test": metrics(labels[test], test_scores, threshold, kinds[test], sources[test]),
        "numpy_runtime_max_abs_error": runtime_max_abs_error,
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
