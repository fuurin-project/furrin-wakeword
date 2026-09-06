#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from oww_distill.model import TinyEmbeddingStudent


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill the openWakeWord embedding encoder")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/student_smoke_4k"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--scheduler", choices=("cosine", "none"), default="cosine",
        help="learning-rate schedule; use 'none' to reproduce the v0 optimizer",
    )
    parser.add_argument("--mse-weight", type=float, default=0.25)
    parser.add_argument(
        "--temporal-delta-weight", type=float, default=0.2,
        help="MSE weight for changes between adjacent 80-ms teacher embeddings",
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=20,
        help="stop after this many epochs without improved validation cosine; 0 disables",
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if args.epochs < 1:
        raise SystemExit("epochs must be positive")
    if args.min_learning_rate < 0 or args.min_learning_rate > args.learning_rate:
        raise SystemExit("min-learning-rate must be between zero and learning-rate")
    if args.mse_weight < 0 or args.temporal_delta_weight < 0:
        raise SystemExit("loss weights must be non-negative")
    if args.early_stopping_patience < 0:
        raise SystemExit("early-stopping-patience must be non-negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with np.load(args.cache, allow_pickle=False) as cache:
        features = cache["features"].astype(np.float32)
        targets = cache["targets"].astype(np.float32)
        splits = cache["splits"].astype(np.uint8)
        sources = cache["sources"].astype(str)
    train_mask = splits == 0
    validation_mask = splits == 1
    if train_mask.sum() < 100 or validation_mask.sum() < 20:
        raise SystemExit("cache does not contain enough train/validation clips")

    # Freeze the raw-feature affine quantizer from training data only.
    raw_features = features
    low, high = np.percentile(raw_features[train_mask], [0.05, 99.95])
    input_scale = float((high - low) / 255.0)
    input_zero_point = int(np.clip(round(-128.0 - low / input_scale), -128, 127))
    clipped_fraction = float(np.mean((raw_features < low) | (raw_features > high)))
    quantized = np.clip(np.rint(raw_features / input_scale) + input_zero_point, -128, 127)
    features = (quantized - input_zero_point) * input_scale
    feature_mean = features[train_mask].mean(axis=(0, 1), keepdims=True)
    feature_std = np.maximum(features[train_mask].std(axis=(0, 1), keepdims=True), 0.05)
    features = (features - feature_mean) / feature_std

    teacher_mean = targets[train_mask].mean(axis=(0, 1), keepdims=True)
    teacher_std = np.maximum(targets[train_mask].std(axis=(0, 1), keepdims=True), 0.25)
    normalized_targets = (targets - teacher_mean) / teacher_std

    def make_dataset(mask: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.from_numpy(features[mask]), torch.from_numpy(normalized_targets[mask])
        )

    train_sources = sources[train_mask]
    names, counts = np.unique(train_sources, return_counts=True)
    inverse = {name: 1.0 / count for name, count in zip(names, counts, strict=True)}
    weights = torch.as_tensor([inverse[name] for name in train_sources], dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights, len(weights), replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(make_dataset(train_mask), args.batch_size, sampler=sampler)
    validation_loader = DataLoader(make_dataset(validation_mask), args.batch_size)
    test_loader = DataLoader(make_dataset(splits == 2), args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyEmbeddingStudent(channels=args.channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_learning_rate
        )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    best = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = train_cosine_sum = train_mse_sum = train_temporal_delta_sum = 0.0
        count = 0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            prediction = model(batch_features)
            cosine = 1.0 - F.cosine_similarity(prediction, batch_targets, dim=2).mean()
            mse = F.mse_loss(prediction, batch_targets)
            temporal_delta = F.mse_loss(
                prediction[:, 1:] - prediction[:, :-1],
                batch_targets[:, 1:] - batch_targets[:, :-1],
            )
            loss = (
                cosine
                + args.mse_weight * mse
                + args.temporal_delta_weight * temporal_delta
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(batch_features)
            train_cosine_sum += float(cosine.detach()) * len(batch_features)
            train_mse_sum += float(mse.detach()) * len(batch_features)
            train_temporal_delta_sum += float(temporal_delta.detach()) * len(batch_features)
            count += len(batch_features)

        model.eval()
        cosine_sum = 0.0
        mse_sum = 0.0
        raw_cosine_sum = 0.0
        temporal_count = 0
        with torch.inference_mode():
            mean_tensor = torch.from_numpy(teacher_mean).to(device)
            std_tensor = torch.from_numpy(teacher_std).to(device)
            for batch_features, batch_targets in validation_loader:
                batch_features = batch_features.to(device, non_blocking=True)
                batch_targets = batch_targets.to(device, non_blocking=True)
                prediction = model(batch_features)
                cosine_sum += float(F.cosine_similarity(prediction, batch_targets, dim=2).sum())
                mse_sum += float(F.mse_loss(prediction, batch_targets, reduction="sum"))
                raw_prediction = prediction * std_tensor + mean_tensor
                raw_target = batch_targets * std_tensor + mean_tensor
                raw_cosine_sum += float(
                    F.cosine_similarity(raw_prediction, raw_target, dim=2).sum()
                )
                temporal_count += len(batch_features) * 16
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss_sum / count,
            "train_cosine_loss": train_cosine_sum / count,
            "train_mse": train_mse_sum / count,
            "train_temporal_delta_mse": train_temporal_delta_sum / count,
            "validation_centered_cosine": cosine_sum / temporal_count,
            "validation_raw_cosine": raw_cosine_sum / temporal_count,
            "validation_normalized_mse": mse_sum / (temporal_count * 96),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["validation_centered_cosine"] > best:
            best = row["validation_centered_cosine"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "architecture": {
                        "feature_bins": 32,
                        "channels": args.channels,
                        "embedding_dim": 96,
                        "deploy_parameters": model.deploy_parameters,
                    },
                    "input_scale": input_scale,
                    "input_zero_point": input_zero_point,
                    "feature_mean": torch.from_numpy(feature_mean.reshape(-1).copy()),
                    "feature_std": torch.from_numpy(feature_std.reshape(-1).copy()),
                    "teacher_mean": torch.from_numpy(teacher_mean.reshape(-1).copy()),
                    "teacher_std": torch.from_numpy(teacher_std.reshape(-1).copy()),
                    "training": {
                        "epochs_requested": args.epochs,
                        "best_epoch": best_epoch,
                        "learning_rate": args.learning_rate,
                        "min_learning_rate": args.min_learning_rate,
                        "scheduler": args.scheduler,
                        "mse_weight": args.mse_weight,
                        "temporal_delta_weight": args.temporal_delta_weight,
                        "seed": args.seed,
                    },
                },
                args.run_dir / "student_best.pt",
            )
        else:
            epochs_without_improvement += 1
        if scheduler is not None:
            scheduler.step()
        if (
            args.early_stopping_patience
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"early stopping at epoch {epoch}; best epoch was {best_epoch}",
                flush=True,
            )
            break

    checkpoint = torch.load(args.run_dir / "student_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])

    def evaluate(loader: DataLoader) -> dict[str, float]:
        model.eval()
        centered_sum = raw_sum = mse_total = 0.0
        frame_count = 0
        mean_tensor = torch.from_numpy(teacher_mean).to(device)
        std_tensor = torch.from_numpy(teacher_std).to(device)
        with torch.inference_mode():
            for batch_features, batch_targets in loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                prediction = model(batch_features)
                centered_sum += float(F.cosine_similarity(prediction, batch_targets, dim=2).sum())
                raw_prediction = prediction * std_tensor + mean_tensor
                raw_target = batch_targets * std_tensor + mean_tensor
                raw_sum += float(F.cosine_similarity(raw_prediction, raw_target, dim=2).sum())
                mse_total += float(F.mse_loss(prediction, batch_targets, reduction="sum"))
                frame_count += len(batch_features) * 16
        return {
            "centered_cosine": centered_sum / frame_count,
            "raw_cosine": raw_sum / frame_count,
            "normalized_mse": mse_total / (frame_count * 96),
        }

    raw_constant = np.broadcast_to(teacher_mean, targets[validation_mask].shape)
    constant_cosine = np.sum(raw_constant * targets[validation_mask], axis=2) / (
        np.linalg.norm(raw_constant, axis=2) * np.linalg.norm(targets[validation_mask], axis=2) + 1e-8
    )
    report = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "cache": str(args.cache),
        "train_clips": int(train_mask.sum()),
        "validation_clips": int(validation_mask.sum()),
        "input_scale": input_scale,
        "input_zero_point": input_zero_point,
        "input_clipped_fraction": clipped_fraction,
        "deploy_parameters": model.deploy_parameters,
        "estimated_int8_weight_bytes": model.deploy_parameters,
        "training_config": {
            "epochs_requested": args.epochs,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "learning_rate": args.learning_rate,
            "min_learning_rate": args.min_learning_rate,
            "scheduler": args.scheduler,
            "mse_weight": args.mse_weight,
            "temporal_delta_weight": args.temporal_delta_weight,
            "early_stopping_patience": args.early_stopping_patience,
            "seed": args.seed,
        },
        "constant_mean_raw_cosine": float(constant_cosine.mean()),
        "best_validation_centered_cosine": best,
        "best_validation": evaluate(validation_loader),
        "held_out_test": evaluate(test_loader),
        "training_seconds": time.monotonic() - started,
        "history": history,
    }
    (args.run_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
