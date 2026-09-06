#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from oww_distill.conv_attention import ConvAttentionHead
from oww_distill.wake_phrase import is_partial_kind


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LiveKit-style Conv-Attention head on frozen student embeddings."
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--wake-name", default="hey_pico")
    parser.add_argument("--layer-dim", type=int, default=16)
    parser.add_argument("--n-blocks", type=int, default=1)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--phase1-epochs", type=int, default=100)
    parser.add_argument("--phase2-epochs", type=int, default=20)
    parser.add_argument("--phase3-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--max-negative-weight", type=float, default=8.0)
    parser.add_argument("--partial-negative-weight", type=float, default=4.0)
    parser.add_argument("--minimum-recall", type=float, default=0.90)
    parser.add_argument("--target-fpph", type=float, default=0.5)
    parser.add_argument("--checkpoint-average", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def focal_loss_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, gamma: float
) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = torch.sigmoid(logits)
    p_t = targets * probabilities + (1.0 - targets) * (1.0 - probabilities)
    return (1.0 - p_t).pow(gamma) * bce


def choose_threshold(labels: np.ndarray, scores: np.ndarray, minimum_recall: float) -> float:
    positives = np.sort(scores[labels == 1])
    allowed_misses = int(np.floor((1.0 - minimum_recall) * len(positives)))
    return float(positives[min(allowed_misses, len(positives) - 1)])


def scores(
    model: nn.Module, values: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    output: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size]).to(device)
            output.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(output)


def evaluate(
    labels: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
    kinds: np.ndarray,
    sources: np.ndarray,
    clip_seconds: float = 2.0,
) -> dict[str, object]:
    positive = labels == 1
    negative = ~positive
    detected = predictions >= threshold
    false_count = int(np.sum(detected & negative))
    negative_hours = float(negative.sum() * clip_seconds / 3600.0)
    return {
        "clips": int(len(labels)),
        "positives": int(positive.sum()),
        "negatives": int(negative.sum()),
        "negative_hours": negative_hours,
        "threshold": threshold,
        "recall": float(np.mean(detected[positive])),
        "false_accept_rate": float(np.mean(detected[negative])),
        "false_accept_count": false_count,
        "estimated_fpph": false_count / negative_hours if negative_hours else 0.0,
        "positive_score_p10": float(np.percentile(predictions[positive], 10)),
        "negative_score_p99": float(np.percentile(predictions[negative], 99)),
        "negative_far_by_kind": {
            kind: float(np.mean(detected[(kinds == kind) & negative]))
            for kind in sorted(set(kinds[negative]))
        },
        "negative_far_by_source": {
            source: float(np.mean(detected[(sources == source) & negative]))
            for source in sorted(set(sources[negative]))
        },
    }


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def main() -> None:
    args = parse_args()
    if args.layer_dim % args.n_heads:
        raise SystemExit("layer-dim must be divisible by n-heads")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise SystemExit("label-smoothing must be in [0, 1)")
    if args.checkpoint_average < 1:
        raise SystemExit("checkpoint-average must be positive")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with np.load(args.cache, allow_pickle=False) as cache:
        values = {name: cache[name] for name in cache.files}
    actual_checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if "student_checkpoint_sha256" in values:
        cached_checkpoint_sha256 = str(values["student_checkpoint_sha256"])
        if cached_checkpoint_sha256 != actual_checkpoint_sha256:
            raise SystemExit(
                "embedding cache and requested student checkpoint do not match: "
                f"{cached_checkpoint_sha256} != {actual_checkpoint_sha256}"
            )
    embeddings = values["embeddings"].astype(np.float32)
    labels = values["labels"].astype(np.float32)
    splits = values["splits"].astype(np.uint8)
    kinds = values["kinds"].astype(str)
    sources = values["sources"].astype(str)
    train, validation, test = splits == 0, splits == 1, splits == 2

    feature_mean = embeddings[train].mean(axis=(0, 1))
    feature_scale = np.maximum(embeddings[train].std(axis=(0, 1)), 1e-5)
    normalized = (embeddings - feature_mean[None, None, :]) / feature_scale[None, None, :]

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    model = ConvAttentionHead(
        layer_dim=args.layer_dim, n_blocks=args.n_blocks, n_heads=args.n_heads
    ).to(device)

    class_counts = np.bincount(labels[train].astype(np.uint8), minlength=2)
    base_weights = len(labels[train]) / (2.0 * class_counts[labels[train].astype(np.uint8)])
    partial = np.asarray([is_partial_kind(kind) for kind in kinds[train]])
    base_weights[partial] *= args.partial_negative_weight
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(normalized[train]),
        torch.from_numpy(labels[train]),
        torch.from_numpy(base_weights.astype(np.float32)),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    history: list[dict[str, float | int]] = []
    candidates: list[dict[str, object]] = []
    phase_specs = [
        (1, args.phase1_epochs, args.learning_rate),
        (2, args.phase2_epochs, args.learning_rate * 0.1),
        (3, args.phase3_epochs, args.learning_rate * 0.01),
    ]
    negative_ceiling = args.max_negative_weight
    global_epoch = 0
    started = time.monotonic()
    for phase, phase_epochs, phase_lr in phase_specs:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=phase_lr, weight_decay=args.weight_decay
        )
        for phase_epoch in range(1, phase_epochs + 1):
            global_epoch += 1
            progress = phase_epoch / max(phase_epochs, 1)
            negative_weight = 1.0 + (negative_ceiling - 1.0) * progress
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            learning_rate = max(phase_lr * cosine_factor, phase_lr * 0.01)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            model.train()
            loss_sum = 0.0
            sample_count = 0
            for batch_x, batch_y, batch_weight in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_weight = batch_weight.to(device)
                batch_weight = torch.where(
                    batch_y < 0.5, batch_weight * negative_weight, batch_weight
                )
                if args.mixup_alpha > 0:
                    lam = torch.distributions.Beta(
                        args.mixup_alpha, args.mixup_alpha
                    ).sample().to(device)
                    permutation = torch.randperm(len(batch_x), device=device)
                    batch_x = lam * batch_x + (1.0 - lam) * batch_x[permutation]
                    batch_y = lam * batch_y + (1.0 - lam) * batch_y[permutation]
                    batch_weight = (
                        lam * batch_weight + (1.0 - lam) * batch_weight[permutation]
                    )
                targets = batch_y * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing
                loss_values = focal_loss_with_logits(model(batch_x), targets, args.focal_gamma)
                loss = (loss_values * batch_weight).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(batch_x)
                sample_count += len(batch_x)

            validation_scores = scores(model, normalized[validation], device, args.batch_size)
            threshold = choose_threshold(
                labels[validation], validation_scores, args.minimum_recall
            )
            validation_metrics = evaluate(
                labels[validation], validation_scores, threshold,
                kinds[validation], sources[validation],
            )
            row = {
                "epoch": global_epoch,
                "phase": phase,
                "phase_epoch": phase_epoch,
                "learning_rate": learning_rate,
                "negative_weight": negative_weight,
                "train_loss": loss_sum / sample_count,
                "validation_recall": float(validation_metrics["recall"]),
                "validation_far": float(validation_metrics["false_accept_rate"]),
                "validation_fpph": float(validation_metrics["estimated_fpph"]),
                "threshold": threshold,
            }
            history.append(row)
            candidates.append(
                {"state": clone_state(model), "metrics": validation_metrics, "row": row}
            )
            print(json.dumps(row), flush=True)

        if float(validation_metrics["estimated_fpph"]) > args.target_fpph:
            negative_ceiling *= 2.0

    candidates.sort(
        key=lambda item: (
            float(item["metrics"]["false_accept_rate"]),
            -float(item["metrics"]["positive_score_p10"]),
        )
    )
    selected = candidates[: min(args.checkpoint_average, len(candidates))]
    averaged_state: dict[str, torch.Tensor] = {}
    for name in selected[0]["state"]:
        averaged_state[name] = torch.stack(
            [candidate["state"][name].float() for candidate in selected]
        ).mean(dim=0)

    best_single_state = copy.deepcopy(candidates[0]["state"])
    choices: list[tuple[str, dict[str, torch.Tensor]]] = [
        ("best_single", best_single_state),
        ("checkpoint_average", averaged_state),
    ]
    best_choice: tuple[str, dict[str, torch.Tensor], dict[str, object], float] | None = None
    for name, state in choices:
        model.load_state_dict(state)
        validation_scores = scores(model, normalized[validation], device, args.batch_size)
        threshold = choose_threshold(labels[validation], validation_scores, args.minimum_recall)
        result = evaluate(
            labels[validation], validation_scores, threshold,
            kinds[validation], sources[validation],
        )
        candidate = (name, state, result, threshold)
        if best_choice is None or float(result["false_accept_rate"]) < float(
            best_choice[2]["false_accept_rate"]
        ):
            best_choice = candidate
    assert best_choice is not None
    selected_name, selected_state, validation_metrics, threshold = best_choice
    model.load_state_dict(selected_state)
    test_scores = scores(model, normalized[test], device, args.batch_size)
    test_metrics = evaluate(
        labels[test], test_scores, threshold, kinds[test], sources[test]
    )

    checkpoint = {
        "format_version": 1,
        "model_state": clone_state(model),
        "architecture": {
            "type": "conv_attention",
            "n_timesteps": 16,
            "embedding_dim": 96,
            "layer_dim": args.layer_dim,
            "n_blocks": args.n_blocks,
            "n_heads": args.n_heads,
            "deploy_parameters": model.deploy_parameters,
        },
        "feature_mean": torch.from_numpy(feature_mean.copy()),
        "feature_scale": torch.from_numpy(feature_scale.copy()),
        "threshold": threshold,
        "name": args.wake_name,
        "student_checkpoint": str(args.checkpoint),
        "student_checkpoint_sha256": actual_checkpoint_sha256,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)

    report = {
        "status": "offline_clip_metrics_not_continuous_or_real_voice",
        "wake_name": args.wake_name,
        "reference": "https://github.com/livekit/livekit-wakeword",
        "backbone": str(args.checkpoint),
        "head": str(args.output),
        "cache": str(args.cache),
        "device": str(device),
        "architecture": checkpoint["architecture"],
        "training": {
            "selected": selected_name,
            "checkpoint_average": len(selected),
            "epochs": global_epoch,
            "seconds": time.monotonic() - started,
            "focal_gamma": args.focal_gamma,
            "mixup_alpha": args.mixup_alpha,
            "label_smoothing": args.label_smoothing,
            "initial_max_negative_weight": args.max_negative_weight,
            "final_max_negative_weight": negative_ceiling,
            "history": history,
        },
        "validation": validation_metrics,
        "test": test_metrics,
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
