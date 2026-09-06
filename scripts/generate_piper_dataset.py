#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
import wave
from pathlib import Path

import numpy as np

from oww_distill.wake_phrase import (
    default_hard_negative_texts,
    default_target_texts,
    negative_kind,
    normalize_phrase,
)


SAMPLE_RATE = 16_000
MAX_SAMPLES = 30_400  # Keep the utterance below the two-second detector window.

TARGET_TEXTS = (
    "Hey Pico",
    "Hey, Pico",
    "hey pico",
    "Hey Pico!",
    "Hey... Pico",
    "Hey Pico?",
)

HARD_NEGATIVE_TEXTS = (
    "Hey",
    "Pico",
    "Hey Pixel",
    "Hey Peter",
    "Hey Peacock",
    "Hey people",
    "Hey pickle",
    "Hey speaker",
    "Hey Peko",
    "Hey Rico",
    "Hey Nico",
    "Hey eco",
    "Okay Pico",
    "Hello Pico",
    "Hi Pico",
    "Hey computer",
    "Hey robot",
    "Okay people",
    "Hello people",
    "Hey, pick up",
    "A pico second",
    "Pico is ready",
    "Where did he go",
    "Here we go",
    "Hey, be cool",
    "Play some music",
    "Stop the music",
    "Turn on the light",
    "What time is it",
    "Set a timer",
    "Open the door",
    "Good morning",
    "Volume up",
    "Volume down",
    "Thank you",
    "Please stop",
    "Pick up the box",
    "Peter picked a pickle",
    "People are speaking",
    "The speaker is on",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a speaker-disjoint Piper Hey Pico dataset and manifest."
    )
    parser.add_argument("--voice-dir", type=Path, default=Path("data/piper/voices"))
    parser.add_argument("--wake-phrase", default="Hey Pico")
    parser.add_argument(
        "--target-text", action="append", dest="target_texts",
        help="positive TTS text; repeat to supply multiple pronunciations",
    )
    parser.add_argument(
        "--hard-negative-text", action="append", dest="hard_negative_texts",
        help="negative or near-phrase TTS text; repeat as needed",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/hey_pico_piper_v1")
    )
    parser.add_argument("--target-per-speaker", type=int, default=24)
    parser.add_argument("--hard-negative-per-speaker", type=int, default=40)
    parser.add_argument("--augmentations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def stable_split(group: str) -> str:
    value = int.from_bytes(hashlib.sha256(group.encode()).digest()[:8], "little") % 100
    return "train" if value < 80 else "validation" if value < 90 else "test"


def resample_linear(audio: np.ndarray, source_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if source_rate == SAMPLE_RATE or len(audio) < 2:
        return audio
    output_length = max(1, int(round(len(audio) * SAMPLE_RATE / source_rate)))
    positions = np.linspace(0, len(audio) - 1, output_length)
    return np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)


def trim(audio: np.ndarray) -> np.ndarray:
    active = np.flatnonzero(np.abs(audio) > 0.0025)
    if active.size:
        margin = int(0.06 * SAMPLE_RATE)
        audio = audio[max(0, int(active[0]) - margin) : min(len(audio), int(active[-1]) + margin)]
    if len(audio) > MAX_SAMPLES:
        positions = np.linspace(0, len(audio) - 1, MAX_SAMPLES)
        audio = np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)
    return audio.astype(np.float32)


def room_effect(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    output = audio.copy()
    for _ in range(int(rng.integers(1, 4))):
        delay = int(rng.uniform(0.012, 0.075) * SAMPLE_RATE)
        if delay < len(audio):
            output[delay:] += float(rng.uniform(0.04, 0.22)) * audio[:-delay]
    return output


def colored_noise(length: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.normal(0.0, 1.0, length).astype(np.float32)
    kind = int(rng.integers(0, 3))
    if kind == 1:
        white = np.cumsum(white).astype(np.float32)
    elif kind == 2:
        white = white - np.concatenate(([0.0], white[:-1]))
    return white / max(float(np.sqrt(np.mean(white * white))), 1e-6)


def augment(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    speed = float(rng.uniform(0.91, 1.10))
    output_length = max(1, int(round(len(audio) / speed)))
    output = np.interp(
        np.linspace(0, len(audio) - 1, output_length), np.arange(len(audio)), audio
    ).astype(np.float32)
    if rng.random() < 0.65:
        output = room_effect(output, rng)
    signal_rms = max(float(np.sqrt(np.mean(output * output))), 1e-5)
    snr_db = float(rng.uniform(10.0, 35.0))
    output += colored_noise(len(output), rng) * signal_rms / (10 ** (snr_db / 20.0))
    output *= float(rng.uniform(0.35, 0.95))
    peak = float(np.max(np.abs(output)))
    if peak > 0.98:
        output *= 0.98 / peak
    return trim(output)


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.rint(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm.tobytes())


def synthesis_audio(voice, text: str, config) -> np.ndarray:
    chunks = list(voice.synthesize(text, syn_config=config))
    if not chunks:
        raise RuntimeError(f"Piper returned no audio for {text!r}")
    sample_rate = int(chunks[0].sample_rate)
    audio = np.concatenate([chunk.audio_float_array for chunk in chunks])
    return trim(resample_linear(audio, sample_rate))


def model_speakers(config_path: Path) -> list[int]:
    config = json.loads(config_path.read_text())
    return list(range(int(config.get("num_speakers", 1))))


def main() -> None:
    args = parse_args()
    try:
        wake_phrase = normalize_phrase(args.wake_phrase)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    default_targets = TARGET_TEXTS if wake_phrase.lower() == "hey pico" else default_target_texts(wake_phrase)
    default_negatives = (
        HARD_NEGATIVE_TEXTS
        if wake_phrase.lower() == "hey pico"
        else default_hard_negative_texts(wake_phrase)
    )
    target_texts = tuple(args.target_texts or default_targets)
    required_partials = default_hard_negative_texts(wake_phrase)[:2]
    requested_negatives = tuple(args.hard_negative_texts or default_negatives)
    hard_negative_texts = tuple(dict.fromkeys((*required_partials, *requested_negatives)))
    if args.target_per_speaker < 1 or args.hard_negative_per_speaker < 1:
        raise SystemExit("per-speaker counts must be positive")
    if args.augmentations < 1:
        raise SystemExit("augmentations must be positive")
    from piper import PiperVoice
    from piper.config import SynthesisConfig

    models = sorted(args.voice_dir.glob("*.onnx"))
    if not models:
        raise SystemExit(f"no Piper .onnx models found in {args.voice_dir}")
    manifest_path = args.output_dir / "manifest.jsonl"
    if manifest_path.exists() and not args.force:
        raise SystemExit(f"{manifest_path} already exists; pass --force to rebuild")
    rng = np.random.default_rng(args.seed)
    records: list[dict[str, object]] = []
    started = time.monotonic()

    for model_index, model_path in enumerate(models, 1):
        config_path = Path(f"{model_path}.json")
        voice_name = model_path.stem
        voice = PiperVoice.load(model_path, config_path=config_path)
        for speaker_id in model_speakers(config_path):
            group = f"{voice_name}:speaker_{speaker_id}"
            split = stable_split(group)
            specifications = []
            for index in range(args.target_per_speaker):
                specifications.append((1, "positive", target_texts[index % len(target_texts)], index))
            for index in range(args.hard_negative_per_speaker):
                text = hard_negative_texts[index % len(hard_negative_texts)]
                specifications.append((0, negative_kind(text, wake_phrase), text, index))

            for label, kind, text, utterance_index in specifications:
                synth_config = SynthesisConfig(
                    speaker_id=speaker_id,
                    length_scale=float(rng.uniform(0.82, 1.20)),
                    noise_scale=float(rng.uniform(0.45, 0.85)),
                    noise_w_scale=float(rng.uniform(0.55, 0.95)),
                    normalize_audio=True,
                    volume=float(rng.uniform(0.75, 1.0)),
                )
                clean = synthesis_audio(voice, text, synth_config)
                text_hash = hashlib.sha1(text.encode()).hexdigest()[:8]
                for augmentation_index in range(args.augmentations):
                    audio = augment(clean, rng)
                    relative = Path("audio") / split / kind / (
                        f"{voice_name}_s{speaker_id:02d}_u{utterance_index:03d}_"
                        f"{text_hash}_a{augmentation_index:02d}.wav"
                    )
                    write_wav(args.output_dir / relative, audio)
                    records.append(
                        {
                            "path": str(relative),
                            "label": label,
                            "kind": kind,
                            "split": split,
                            "group": group,
                            "source": "piper",
                            "voice": voice_name,
                            "speaker_id": speaker_id,
                            "text": text,
                            "augmentation": augmentation_index,
                            "wake_phrase": wake_phrase,
                        }
                    )
        print(
            f"model {model_index}/{len(models)} {voice_name}: records={len(records)}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    summary = {
        "format_version": 1,
        "generator": "piper-tts",
        "wake_phrase": wake_phrase,
        "target_texts": list(target_texts),
        "hard_negative_texts": list(hard_negative_texts),
        "voice_source": "https://huggingface.co/rhasspy/piper-voices",
        "seed": args.seed,
        "models": [path.name for path in models],
        "speaker_groups": len({str(record["group"]) for record in records}),
        "records": len(records),
        "by_split": {
            split: sum(record["split"] == split for record in records)
            for split in ("train", "validation", "test")
        },
        "by_kind": {
            kind: sum(record["kind"] == kind for record in records)
            for kind in sorted({str(record["kind"]) for record in records})
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
