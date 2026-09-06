#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from oww_distill.classifier import DistilledStudentRuntime, TemporalWakeClassifier
from oww_distill.model import TinyEmbeddingStudent
from train_temporal_hey_pico_head import (
    TemporalHead,
    choose_threshold,
    deterministic_window,
    metrics,
    model_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Jointly fine-tune the final student blocks and temporal Hey Pico head "
            "while preserving generic teacher embeddings."
        )
    )
    parser.add_argument(
        "--wake-cache", type=Path,
        default=Path("data/cache/hey_pico_piper_open_speech_student.npz"),
    )
    parser.add_argument(
        "--wake-feature-cache", type=Path,
        default=Path("data/cache/hey_pico_piper_open_speech_features.npz"),
    )
    parser.add_argument(
        "--generic-cache", type=Path, default=Path("data/cache/distill_balanced_57078.npz")
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("runs/student_balanced_57078/student_best.pt"),
    )
    parser.add_argument(
        "--constants", type=Path, default=Path("artifacts/fuurin_frontend.npz")
    )
    parser.add_argument(
        "--head", type=Path,
        default=Path("artifacts/hey_pico_temporal_piper_v1_head.npz"),
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("runs/student_hey_pico_finetune_v1")
    )
    parser.add_argument(
        "--head-output", type=Path,
        default=Path("artifacts/hey_pico_finetuned_candidate.npz"),
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--preservation-weight", type=float, default=0.15)
    parser.add_argument("--max-centered-cosine-drop", type=float, default=0.03)
    parser.add_argument("--partial-negative-weight", type=float, default=4.0)
    parser.add_argument("--minimum-recall", type=float, default=0.90)
    parser.add_argument("--unfreeze-blocks", type=int, default=2)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--rebuild-wake-features", action="store_true")
    return parser.parse_args()


def build_wake_features(args: argparse.Namespace, runtime: DistilledStudentRuntime) -> np.ndarray:
    with np.load(args.wake_cache, allow_pickle=False) as cache:
        paths = cache["paths"].astype(str)
    parts = []
    pending = []
    started = time.monotonic()
    for index, path in enumerate(paths, 1):
        pending.append(runtime.frontend.streaming_clip(deterministic_window(Path(path))))
        if len(pending) >= args.batch_size or index == len(paths):
            raw = np.stack(pending).astype(np.float32)
            quantized = np.clip(
                np.rint(raw / runtime.input_scale) + runtime.input_zero_point, -128, 127
            )
            dequantized = (quantized - runtime.input_zero_point) * runtime.input_scale
            mean = runtime.feature_mean.cpu().numpy()
            std = runtime.feature_std.cpu().numpy()
            parts.append(((dequantized - mean) / std).astype(np.float16))
            pending.clear()
        if index == 1 or index % 1000 == 0 or index == len(paths):
            elapsed = max(time.monotonic() - started, 1e-6)
            print(f"wake features {index}/{len(paths)} ({index / elapsed:.1f} clips/s)", flush=True)
    features = np.concatenate(parts)
    args.wake_feature_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.wake_feature_cache, format_version=np.array(1), features=features)
    return features


def load_wake_features(args: argparse.Namespace, runtime: DistilledStudentRuntime) -> np.ndarray:
    if args.rebuild_wake_features or not args.wake_feature_cache.exists():
        return build_wake_features(args, runtime).astype(np.float32)
    with np.load(args.wake_feature_cache, allow_pickle=False) as cache:
        return cache["features"].astype(np.float32)


def load_head(path: Path, device: torch.device) -> tuple[TemporalHead, TemporalWakeClassifier]:
    saved = TemporalWakeClassifier.load(path)
    model = TemporalHead(
        input_dim=len(saved.feature_mean), channels=len(saved.pointwise_bias),
        kernel_size=saved.depthwise_weights.shape[1],
    ).to(device)
    with torch.no_grad():
        model.depthwise.weight.copy_(torch.from_numpy(saved.depthwise_weights[:, None, :]))
        model.depthwise.bias.copy_(torch.from_numpy(saved.depthwise_bias))
        model.pointwise.weight.copy_(torch.from_numpy(saved.pointwise_weights[:, :, None]))
        model.pointwise.bias.copy_(torch.from_numpy(saved.pointwise_bias))
        model.output.weight.copy_(torch.from_numpy(saved.output_weights[None, :]))
        model.output.bias.copy_(torch.tensor([saved.output_bias]))
    return model, saved


def generic_metrics(
    model: TinyEmbeddingStudent, features: np.ndarray, targets: np.ndarray,
    device: torch.device, batch_size: int,
) -> dict[str, float]:
    centered_sum = mse_sum = 0.0
    frame_count = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            x = torch.from_numpy(features[start : start + batch_size]).to(device)
            y = torch.from_numpy(targets[start : start + batch_size]).to(device)
            prediction = model(x)
            centered_sum += float(nn.functional.cosine_similarity(prediction, y, dim=2).sum())
            mse_sum += float(nn.functional.mse_loss(prediction, y, reduction="sum"))
            frame_count += len(x) * prediction.shape[1]
    return {
        "centered_cosine": centered_sum / frame_count,
        "normalized_mse": mse_sum / (frame_count * targets.shape[2]),
    }


def save_pair(
    backbone: TinyEmbeddingStudent, head: TemporalHead, saved_head: TemporalWakeClassifier,
    original_checkpoint: dict, threshold: float, checkpoint_output: Path,
    head_output: Path,
) -> None:
    checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    output_checkpoint = {
        key: value for key, value in original_checkpoint.items() if key != "model_state"
    }
    output_checkpoint["model_state"] = {
        name: value.detach().cpu().clone() for name, value in backbone.state_dict().items()
    }
    output_checkpoint["fine_tuned_for"] = "hey_pico_piper_v1"
    torch.save(output_checkpoint, checkpoint_output)
    TemporalWakeClassifier(
        depthwise_weights=head.depthwise.weight.detach().cpu().numpy()[:, 0, :],
        depthwise_bias=head.depthwise.bias.detach().cpu().numpy(),
        pointwise_weights=head.pointwise.weight.detach().cpu().numpy()[:, :, 0],
        pointwise_bias=head.pointwise.bias.detach().cpu().numpy(),
        output_weights=head.output.weight.detach().cpu().numpy()[0],
        output_bias=float(head.output.bias.detach().cpu().numpy()[0]),
        feature_mean=saved_head.feature_mean,
        feature_scale=saved_head.feature_scale,
        threshold=threshold,
    ).save(head_output)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    runtime = DistilledStudentRuntime(args.checkpoint, args.constants, str(device))
    wake_features = load_wake_features(args, runtime)
    with np.load(args.wake_cache, allow_pickle=False) as cache:
        labels = cache["labels"].astype(np.float32)
        splits = cache["splits"].astype(np.uint8)
        kinds = cache["kinds"].astype(str)
        sources = cache["sources"].astype(str)
    with np.load(args.generic_cache, allow_pickle=False) as cache:
        generic_raw = cache["features"].astype(np.float32)
        generic_targets_raw = cache["targets"].astype(np.float32)
        generic_splits = cache["splits"].astype(np.uint8)
    original = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    input_scale = float(original["input_scale"])
    input_zero_point = int(original["input_zero_point"])
    quantized = np.clip(np.rint(generic_raw / input_scale) + input_zero_point, -128, 127)
    generic_features = (quantized - input_zero_point) * input_scale
    generic_features = (
        generic_features - original["feature_mean"].numpy()[None, None, :]
    ) / original["feature_std"].numpy()[None, None, :]
    generic_targets = (
        generic_targets_raw - original["teacher_mean"].numpy()[None, None, :]
    ) / original["teacher_std"].numpy()[None, None, :]
    del generic_raw, generic_targets_raw, quantized

    backbone = TinyEmbeddingStudent(
        feature_bins=int(original["architecture"]["feature_bins"]),
        channels=int(original["architecture"]["channels"]),
    ).to(device)
    backbone.load_state_dict(original["model_state"])
    head, saved_head = load_head(args.head, device)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    for block in backbone.blocks[-args.unfreeze_blocks :]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for parameter in backbone.output_projection.parameters():
        parameter.requires_grad = True
    trainable = [parameter for parameter in backbone.parameters() if parameter.requires_grad]
    trainable += list(head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)

    wake_train, wake_validation, wake_test = splits == 0, splits == 1, splits == 2
    wake_mean = saved_head.feature_mean[None, None, :]
    wake_scale = saved_head.feature_scale[None, None, :]
    counts = np.bincount(labels[wake_train].astype(np.uint8), minlength=2)
    wake_weights = len(labels[wake_train]) / (2.0 * counts[labels[wake_train].astype(np.uint8)])
    wake_weights[np.isin(kinds[wake_train], ("hey_only", "pico_only"))] *= args.partial_negative_weight
    wake_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(wake_features[wake_train]),
        torch.from_numpy(labels[wake_train]),
        torch.from_numpy(wake_weights.astype(np.float32)),
    )
    generic_train = generic_splits == 0
    generic_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(generic_features[generic_train].astype(np.float32)),
        torch.from_numpy(generic_targets[generic_train].astype(np.float32)),
    )
    wake_loader = torch.utils.data.DataLoader(
        wake_dataset, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    generic_loader = torch.utils.data.DataLoader(
        generic_dataset, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    generic_validation = generic_splits == 1
    baseline_generic = generic_metrics(
        backbone, generic_features[generic_validation], generic_targets[generic_validation],
        device, args.batch_size,
    )

    def wake_scores(mask: np.ndarray) -> np.ndarray:
        backbone.eval()
        head.eval()
        outputs = []
        indices = np.flatnonzero(mask)
        with torch.inference_mode():
            for start in range(0, len(indices), args.batch_size):
                x = torch.from_numpy(wake_features[indices[start : start + args.batch_size]]).to(device)
                embedding = backbone(x).cpu().numpy()
                normalized = (embedding - wake_mean) / wake_scale
                outputs.append(model_scores(head, normalized.astype(np.float32), device, args.batch_size))
        return np.concatenate(outputs)

    initial_val_scores = wake_scores(wake_validation)
    initial_threshold = choose_threshold(labels[wake_validation], initial_val_scores, args.minimum_recall)
    initial_far = float(np.mean(initial_val_scores[labels[wake_validation] == 0] >= initial_threshold))
    best = {
        "epoch": 0, "far": initial_far, "threshold": initial_threshold,
        "backbone": copy.deepcopy(backbone.state_dict()), "head": copy.deepcopy(head.state_dict()),
        "generic": baseline_generic,
    }
    stale = 0
    history = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        backbone.train()
        head.train()
        generic_iterator = iter(generic_loader)
        losses = []
        for wake_x, wake_y, wake_weight in wake_loader:
            try:
                generic_x, generic_y = next(generic_iterator)
            except StopIteration:
                generic_iterator = iter(generic_loader)
                generic_x, generic_y = next(generic_iterator)
            wake_x, wake_y, wake_weight = wake_x.to(device), wake_y.to(device), wake_weight.to(device)
            generic_x, generic_y = generic_x.to(device), generic_y.to(device)
            wake_embedding = backbone(wake_x)
            normalized_wake = (
                wake_embedding - torch.from_numpy(wake_mean).to(device)
            ) / torch.from_numpy(wake_scale).to(device)
            wake_loss = nn.functional.binary_cross_entropy_with_logits(
                head(normalized_wake), wake_y, weight=wake_weight,
            )
            generic_prediction = backbone(generic_x)
            preserve_cosine = 1.0 - nn.functional.cosine_similarity(
                generic_prediction, generic_y, dim=2
            ).mean()
            preserve_mse = nn.functional.mse_loss(generic_prediction, generic_y)
            preservation_loss = preserve_cosine + 0.25 * preserve_mse
            loss = wake_loss + args.preservation_weight * preservation_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append((float(loss.detach()), float(wake_loss.detach()), float(preservation_loss.detach())))
        val_scores = wake_scores(wake_validation)
        threshold = choose_threshold(labels[wake_validation], val_scores, args.minimum_recall)
        far = float(np.mean(val_scores[labels[wake_validation] == 0] >= threshold))
        generic_validation_metrics = generic_metrics(
            backbone, generic_features[generic_validation], generic_targets[generic_validation],
            device, args.batch_size,
        )
        allowed = generic_validation_metrics["centered_cosine"] >= (
            baseline_generic["centered_cosine"] - args.max_centered_cosine_drop
        )
        row = {
            "epoch": epoch, "loss": float(np.mean([x[0] for x in losses])),
            "wake_loss": float(np.mean([x[1] for x in losses])),
            "preservation_loss": float(np.mean([x[2] for x in losses])),
            "validation_far": far, "threshold": threshold,
            "generic_validation_centered_cosine": generic_validation_metrics["centered_cosine"],
            "eligible": allowed,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if allowed and far < best["far"]:
            best = {
                "epoch": epoch, "far": far, "threshold": threshold,
                "backbone": copy.deepcopy(backbone.state_dict()),
                "head": copy.deepcopy(head.state_dict()),
                "generic": generic_validation_metrics,
            }
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break

    backbone.load_state_dict(best["backbone"])
    head.load_state_dict(best["head"])
    val_scores = wake_scores(wake_validation)
    test_scores = wake_scores(wake_test)
    threshold = choose_threshold(labels[wake_validation], val_scores, args.minimum_recall)
    generic_test = generic_metrics(
        backbone, generic_features[generic_splits == 2], generic_targets[generic_splits == 2],
        device, args.batch_size,
    )
    checkpoint_output = args.run_dir / "student_best.pt"
    save_pair(backbone, head, saved_head, original, threshold, checkpoint_output, args.head_output)
    trainable_backbone_parameters = sum(
        parameter.numel() for parameter in backbone.parameters() if parameter.requires_grad
    )
    report = {
        "status": "joint_finetune_offline_piper_and_open_speech_not_real_voice_or_continuous_evidence",
        "base_checkpoint": str(args.checkpoint), "checkpoint": str(checkpoint_output),
        "head": str(args.head_output), "paired_artifacts_required": True,
        "device": str(device), "best_epoch": int(best["epoch"]),
        "epochs_ran": len(history), "training_seconds": time.monotonic() - started,
        "trainable_backbone_parameters": trainable_backbone_parameters,
        "head_parameters": sum(parameter.numel() for parameter in head.parameters()),
        "baseline_generic_validation": baseline_generic,
        "selected_generic_validation": best["generic"], "selected_generic_test": generic_test,
        "validation": metrics(labels[wake_validation], val_scores, threshold, kinds[wake_validation], sources[wake_validation]),
        "test": metrics(labels[wake_test], test_scores, threshold, kinds[wake_test], sources[wake_test]),
        "history": history,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
