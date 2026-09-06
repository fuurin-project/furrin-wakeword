#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from oww_distill.classifier import DistilledStudentRuntime


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-embed an existing aligned feature cache with a new generic student."
    )
    parser.add_argument(
        "--features-cache",
        type=Path,
        default=Path("data/cache/hey_pico_piper_open_speech_features.npz"),
    )
    parser.add_argument(
        "--metadata-cache",
        type=Path,
        default=Path("data/cache/hey_pico_piper_open_speech_student.npz"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--constants", type=Path, default=Path("artifacts/fuurin_frontend.npz")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--alignment", choices=("cached", "livekit", "streaming"), default="cached",
        help="reuse cached features or rebuild windows with positives aligned near the end",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("batch-size must be positive")

    with np.load(args.metadata_cache, allow_pickle=False) as values:
        metadata = {
            name: values[name]
            for name in ("labels", "splits", "kinds", "sources", "groups", "paths")
        }
    features = None
    if args.alignment == "cached":
        with np.load(args.features_cache, allow_pickle=False) as values:
            features = values["features"]
        if len(features) != len(metadata["labels"]):
            raise SystemExit("feature and metadata cache lengths do not match")

    runtime = DistilledStudentRuntime(args.checkpoint, args.constants, args.device)
    output: list[np.ndarray] = []
    pending: list[np.ndarray] = []

    def flush() -> None:
        if pending:
            output.append(runtime.embed_feature_batch(np.stack(pending)).astype(np.float16))
            pending.clear()

    if args.alignment == "cached":
        assert features is not None
        for start in range(0, len(features), args.batch_size):
            stop = min(start + args.batch_size, len(features))
            output.append(runtime.embed_feature_batch(features[start:stop]).astype(np.float16))
            if start == 0 or stop == len(features) or stop % 4096 < args.batch_size:
                print(f"embed {stop}/{len(features)}", flush=True)
    else:
        for index, (path_value, label_value, kind_value) in enumerate(
            zip(
                metadata["paths"].astype(str),
                metadata["labels"],
                metadata["kinds"].astype(str),
                strict=True,
            ),
            1,
        ):
            path = Path(path_value)
            audio, sample_rate = sf.read(path, dtype="int16", always_2d=False)
            if audio.ndim == 2:
                audio = np.rint(audio.astype(np.float32).mean(axis=1)).astype(np.int16)
            audio = np.asarray(audio, dtype=np.int16).reshape(-1)
            if sample_rate != 16_000:
                positions = np.linspace(
                    0, len(audio) - 1,
                    max(1, int(round(len(audio) * 16_000 / sample_rate))),
                )
                audio = np.interp(positions, np.arange(len(audio)), audio).astype(np.int16)
            window = np.zeros(32_000, dtype=np.int16)
            # LiveKit end-aligns positives. For continuous streaming evaluation,
            # speech negatives need the same alignment distribution so the head
            # cannot solve the task from position alone.
            align_to_end = int(label_value) == 1 or (
                args.alignment == "streaming" and kind_value != "noise"
            )
            if align_to_end:
                seed = int.from_bytes(hashlib.sha256(path_value.encode()).digest()[:8], "little")
                jitter = seed % 3201
                end = 32_000 - jitter
                start = max(0, end - len(audio))
                source_start = max(0, len(audio) - (end - start))
                window[start:end] = audio[source_start : source_start + end - start]
            elif len(audio) >= 32_000:
                start = (len(audio) - 32_000) // 2
                window = audio[start : start + 32_000].copy()
            else:
                start = (32_000 - len(audio)) // 2
                window[start : start + len(audio)] = audio
            pending.append(runtime.frontend.streaming_clip(window))
            if len(pending) >= args.batch_size:
                flush()
            if index == 1 or index == len(metadata["labels"]) or index % 4096 == 0:
                print(f"embed {index}/{len(metadata['labels'])}", flush=True)
        flush()

    checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        format_version=np.array(1, dtype=np.int32),
        embeddings=np.concatenate(output),
        **metadata,
        alignment=np.array(args.alignment),
        student_checkpoint=np.array(str(args.checkpoint)),
        student_checkpoint_sha256=np.array(checkpoint_sha256),
    )
    print(
        f"saved {args.output}: clips={len(metadata['labels'])} "
        f"alignment={args.alignment} checkpoint={checkpoint_sha256}"
    )


if __name__ == "__main__":
    main()
