#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from oww_distill.frontend import DspFrontend, FrontendConstants, SAMPLE_RATE


CLIP_SAMPLES = 2 * SAMPLE_RATE


def split_hash(group: str) -> int:
    value = int.from_bytes(hashlib.sha256(group.encode()).digest()[:8], "little") % 100
    return 0 if value < 80 else 1 if value < 90 else 2


def center_clip(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.floating):
        audio = np.clip(np.rint(audio * 32768), -32768, 32767).astype(np.int16)
    else:
        audio = audio.astype(np.int16)
    if audio.size >= CLIP_SAMPLES:
        start = (audio.size - CLIP_SAMPLES) // 2
        return audio[start : start + CLIP_SAMPLES].copy()
    output = np.zeros(CLIP_SAMPLES, dtype=np.int16)
    start = (CLIP_SAMPLES - audio.size) // 2
    output[start : start + audio.size] = audio
    return output


def choose(paths: list[Path], count: int, seed: int) -> list[Path]:
    paths = sorted(paths)
    rng = np.random.default_rng(seed)
    if count and len(paths) > count:
        return [paths[index] for index in sorted(rng.choice(len(paths), count, replace=False))]
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache aligned DSP features and openWakeWord targets")
    parser.add_argument("--librispeech", type=Path, required=True)
    parser.add_argument("--speech-commands", type=Path, required=True)
    parser.add_argument("--constants", type=Path, default=Path("artifacts/fuurin_frontend.npz"))
    parser.add_argument("--output", type=Path, default=Path("data/cache/distill_smoke_4k.npz"))
    parser.add_argument("--max-librispeech", type=int, default=2000)
    parser.add_argument("--max-speech-commands", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--ncpu", type=int, default=4)
    args = parser.parse_args()

    libri = choose(list(args.librispeech.rglob("*.flac")), args.max_librispeech, args.seed)
    commands = choose(
        [p for p in args.speech_commands.rglob("*.wav") if not p.parent.name.startswith("_")],
        args.max_speech_commands,
        args.seed + 1,
    )
    rows = [(path, "librispeech") for path in libri] + [
        (path, "speech_commands") for path in commands
    ]
    if not rows:
        raise SystemExit("no audio files discovered")

    os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
    from openwakeword.utils import AudioFeatures

    teacher = AudioFeatures(inference_framework="tflite", ncpu=args.ncpu)
    frontend = DspFrontend(FrontendConstants.load(args.constants))
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    splits: list[int] = []
    sources: list[str] = []
    paths_out: list[str] = []
    started = time.monotonic()
    for index, (path, source) in enumerate(rows, 1):
        try:
            audio, sample_rate = sf.read(path, always_2d=False)
            if sample_rate != SAMPLE_RATE:
                raise ValueError(f"sample rate {sample_rate}")
            pcm = center_clip(audio)
            feature = frontend.streaming_clip(pcm)
            target = np.asarray(teacher._get_embeddings(pcm), dtype=np.float32)
            if feature.shape != (197, 32) or target.shape != (16, 96):
                raise ValueError(f"shape feature={feature.shape}, target={target.shape}")
            if source == "librispeech":
                group = path.relative_to(args.librispeech).parts[0]
            else:
                group = path.stem.split("_nohash_")[0]
            features.append(feature.astype(np.float16))
            targets.append(target.astype(np.float16))
            splits.append(split_hash(f"{source}:{group}"))
            sources.append(source)
            paths_out.append(str(path))
        except Exception as exc:
            print(f"skip {path}: {exc}", flush=True)
        if index == 1 or index % 100 == 0 or index == len(rows):
            rate = index / max(time.monotonic() - started, 1e-6)
            print(f"cache {index}/{len(rows)} ({rate:.2f} clips/s)", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        format_version=np.array(1, dtype=np.int32),
        features=np.stack(features),
        targets=np.stack(targets),
        splits=np.asarray(splits, dtype=np.uint8),
        sources=np.asarray(sources),
        paths=np.asarray(paths_out),
    )
    counts = np.bincount(np.asarray(splits), minlength=3)
    print(f"saved {args.output}: clips={len(features)}, train/val/test={counts.tolist()}")


if __name__ == "__main__":
    main()
