#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from oww_distill.classifier import LinearCommandClassifier
from oww_distill.model import TinyEmbeddingStudent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a matched frozen linear probe on Speech Commands using teacher, "
            "distilled-student, and random-student embeddings."
        )
    )
    parser.add_argument(
        "--cache", type=Path, default=Path("data/cache/distill_balanced_57078.npz")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/student_balanced_57078/student_best.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/speech_commands_linear_probe/metrics.json"),
    )
    parser.add_argument(
        "--student-head-output",
        type=Path,
        default=Path("artifacts/speech_commands_student_head.npz"),
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--regularization", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but CUDA is unavailable")
    return torch.device(name)


def student_embeddings(
    features: np.ndarray,
    checkpoint: dict,
    batch_size: int,
    device: torch.device,
    *,
    random_weights: bool,
    seed: int,
) -> np.ndarray:
    architecture = checkpoint["architecture"]
    if random_weights:
        torch.manual_seed(seed)
    model = TinyEmbeddingStudent(
        feature_bins=int(architecture["feature_bins"]),
        channels=int(architecture["channels"]),
    ).to(device)
    if not random_weights:
        model.load_state_dict(checkpoint["model_state"])
    model.eval()

    input_scale = float(checkpoint["input_scale"])
    input_zero_point = int(checkpoint["input_zero_point"])
    feature_mean = checkpoint["feature_mean"].cpu().numpy().reshape(1, 1, -1)
    feature_std = checkpoint["feature_std"].cpu().numpy().reshape(1, 1, -1)
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = features[start : start + batch_size].astype(np.float32)
            quantized = np.clip(
                np.rint(batch / input_scale) + input_zero_point, -128, 127
            )
            batch = (quantized - input_zero_point) * input_scale
            batch = (batch - feature_mean) / feature_std
            prediction = model(torch.from_numpy(batch).to(device))
            outputs.append(prediction.mean(dim=1).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def top_k_accuracy(probabilities: np.ndarray, labels: np.ndarray, k: int) -> float:
    top_k = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    return float(np.mean(np.any(top_k == labels[:, None], axis=1)))


def evaluate(
    name: str,
    embeddings: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    class_names: list[str],
    max_iter: int,
    regularization: float,
    seed: int,
    head_output: Path | None = None,
) -> dict:
    train = splits == 0
    validation = splits == 1
    test = splits == 2
    head = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=regularization,
            max_iter=max_iter,
            solver="lbfgs",
            random_state=seed,
        ),
    )
    head.fit(embeddings[train], labels[train])
    if head_output is not None:
        LinearCommandClassifier(
            weights=head[-1].coef_,
            bias=head[-1].intercept_,
            feature_mean=head[0].mean_,
            feature_scale=head[0].scale_,
            classes=np.asarray(class_names),
        ).save(head_output)

    report: dict[str, object] = {
        "name": name,
        "embedding_dim": int(embeddings.shape[1]),
        "pooling": "temporal_mean",
        "head": "standard_scaler_plus_multiclass_logistic_regression",
        "head_parameters": int(embeddings.shape[1] * len(class_names) + len(class_names)),
        "iterations": [int(value) for value in head[-1].n_iter_],
    }
    for split_name, mask in (("validation", validation), ("test", test)):
        probabilities = head.predict_proba(embeddings[mask])
        predictions = np.argmax(probabilities, axis=1)
        per_class = {}
        for index, class_name in enumerate(class_names):
            class_mask = labels[mask] == index
            per_class[class_name] = float(np.mean(predictions[class_mask] == index))
        report[split_name] = {
            "clips": int(mask.sum()),
            "accuracy": float(accuracy_score(labels[mask], predictions)),
            "balanced_accuracy": float(
                balanced_accuracy_score(labels[mask], predictions)
            ),
            "macro_f1": float(f1_score(labels[mask], predictions, average="macro")),
            "top5_accuracy": top_k_accuracy(probabilities, labels[mask], 5),
            "per_class_recall": per_class,
        }
    return report


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.max_iter < 1 or args.regularization <= 0:
        raise SystemExit("batch size, max iterations, and regularization must be positive")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    with np.load(args.cache, allow_pickle=False) as cache:
        source_mask = cache["sources"].astype(str) == "speech_commands"
        features = cache["features"][source_mask]
        teacher = cache["targets"][source_mask].astype(np.float32).mean(axis=1)
        splits = cache["splits"][source_mask].astype(np.uint8)
        paths = cache["paths"][source_mask].astype(str)

    class_names = sorted({Path(path).parent.name for path in paths})
    class_to_index = {name: index for index, name in enumerate(class_names)}
    labels = np.asarray(
        [class_to_index[Path(path).parent.name] for path in paths], dtype=np.int64
    )
    if set(np.unique(splits)) != {0, 1, 2}:
        raise SystemExit("Speech Commands cache must contain train, validation, and test")
    if len(class_names) < 2:
        raise SystemExit("Speech Commands cache must contain at least two classes")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    distilled = student_embeddings(
        features, checkpoint, args.batch_size, device, random_weights=False, seed=args.seed
    )
    random_student = student_embeddings(
        features, checkpoint, args.batch_size, device, random_weights=True, seed=args.seed
    )

    common = {
        "cache": str(args.cache),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "seed": args.seed,
        "classes": class_names,
        "class_count": len(class_names),
        "split_clips": {
            "train": int(np.sum(splits == 0)),
            "validation": int(np.sum(splits == 1)),
            "test": int(np.sum(splits == 2)),
        },
        "split_policy": "existing source-speaker hash split from teacher cache",
        "dataset_fingerprint": hashlib.sha256(
            "\n".join(f"{path}\t{split}" for path, split in zip(paths, splits, strict=True)).encode()
        ).hexdigest(),
        "probe": {
            "pooling": "temporal_mean",
            "regularization_C": args.regularization,
            "max_iter": args.max_iter,
            "selection": "fixed configuration; validation is reported but not used for tuning",
        },
    }
    results = [
        evaluate(
            "teacher", teacher, labels, splits, class_names,
            args.max_iter, args.regularization, args.seed,
        ),
        evaluate(
            "distilled_student", distilled, labels, splits, class_names,
            args.max_iter, args.regularization, args.seed, args.student_head_output,
        ),
        evaluate(
            "random_student", random_student, labels, splits, class_names,
            args.max_iter, args.regularization, args.seed,
        ),
    ]
    report = {**common, "results": results}
    report["student_head_output"] = str(args.student_head_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
