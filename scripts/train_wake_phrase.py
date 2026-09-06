#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from oww_distill.wake_phrase import normalize_phrase, phrase_slug


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a voice-disjoint Piper dataset and train reusable wake-word "
            "heads on the frozen generic student backbone."
        )
    )
    parser.add_argument("--wake-phrase", required=True, help='for example: "Hey Nordic"')
    parser.add_argument("--name", help="artifact slug; defaults to the normalized phrase")
    parser.add_argument(
        "--head", choices=("temporal", "both"), default="temporal",
        help="train the MCU-sized temporal head, or it plus Conv-Attention",
    )
    parser.add_argument("--voice-dir", type=Path, default=Path("data/piper/voices"))
    parser.add_argument(
        "--piper-python", type=Path,
        help="Python containing piper-tts; auto-detected when omitted",
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
        default=Path("artifacts/fuurin_embed.pt"),
    )
    parser.add_argument(
        "--constants", type=Path, default=Path("artifacts/fuurin_frontend.npz")
    )
    parser.add_argument("--target-text", action="append", dest="target_texts")
    parser.add_argument("--hard-negative-text", action="append", dest="hard_negative_texts")
    parser.add_argument("--target-per-speaker", type=int, default=24)
    parser.add_argument("--hard-negative-per-speaker", type=int, default=40)
    parser.add_argument("--augmentations", type=int, default=3)
    parser.add_argument("--minimum-recall", type=float, default=0.90)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force-dataset", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def run(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def has_piper(python: Path) -> bool:
    if not python.exists():
        return False
    return subprocess.run(
        [str(python), "-c", "import piper"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def find_piper_python(requested: Path | None) -> Path:
    if requested is not None:
        candidate = requested.expanduser().resolve()
        if not has_piper(candidate):
            raise SystemExit(f"piper-tts is unavailable in {candidate}")
        return candidate
    candidates = [
        Path(sys.executable),
        Path("/home/user/Workspace/fuurin-wake/.venv/bin/python"),
    ]
    for candidate in candidates:
        if has_piper(candidate):
            return candidate
    raise SystemExit(
        "piper-tts is not installed; install the data extra or pass "
        "--piper-python /path/to/python"
    )


def main() -> None:
    args = parse_args()
    try:
        phrase = normalize_phrase(args.wake_phrase)
        name = phrase_slug(args.name or phrase)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    dataset_dir = ROOT / "data" / f"{name}_piper_v1"
    manifest = dataset_dir / "manifest.jsonl"
    cache = ROOT / "data" / "cache" / f"{name}_piper_open_speech_streaming.npz"
    temporal_head = ROOT / "artifacts" / f"{name}_temporal_head.npz"
    temporal_metrics = ROOT / "runs" / f"{name}_temporal" / "metrics.json"
    attention_head = ROOT / "artifacts" / f"{name}_conv_attention_head.pt"
    attention_metrics = ROOT / "runs" / f"{name}_conv_attention" / "metrics.json"
    checkpoint = resolve(ROOT, args.checkpoint)
    constants = resolve(ROOT, args.constants)
    voice_dir = resolve(ROOT, args.voice_dir)

    env = os.environ.copy()
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    python = sys.executable
    piper_python = find_piper_python(args.piper_python)

    if args.force_dataset or not manifest.exists():
        command = [
            str(piper_python), "scripts/generate_piper_dataset.py",
            "--wake-phrase", phrase,
            "--voice-dir", str(voice_dir),
            "--output-dir", str(dataset_dir),
            "--target-per-speaker", str(args.target_per_speaker),
            "--hard-negative-per-speaker", str(args.hard_negative_per_speaker),
            "--augmentations", str(args.augmentations),
        ]
        if args.force_dataset:
            command.append("--force")
        for text in args.target_texts or ():
            command.extend(("--target-text", text))
        for text in args.hard_negative_texts or ():
            command.extend(("--hard-negative-text", text))
        run(command, env, args.dry_run)
    else:
        print(f"reuse dataset: {manifest}", flush=True)

    temporal_command = [
        python, "scripts/train_temporal_hey_pico_head.py",
        "--wake-name", name,
        "--manifest", str(manifest),
        "--external-root", str(args.external_root),
        "--edge-hard-negative-dir", str(args.edge_hard_negative_dir),
        "--edge-noise-dir", str(args.edge_noise_dir),
        "--checkpoint", str(checkpoint),
        "--constants", str(constants),
        "--cache", str(cache),
        "--output", str(temporal_head),
        "--metrics", str(temporal_metrics),
        "--alignment", "streaming",
        "--minimum-recall", str(args.minimum_recall),
        "--device", args.device,
    ]
    if args.rebuild_cache or args.force_dataset:
        temporal_command.append("--rebuild-cache")
    run(temporal_command, env, args.dry_run)

    if args.head == "both":
        run(
            [
                python, "scripts/train_conv_attention_hey_pico_head.py",
                "--wake-name", name,
                "--cache", str(cache),
                "--checkpoint", str(checkpoint),
                "--output", str(attention_head),
                "--metrics", str(attention_metrics),
                "--layer-dim", "32",
                "--n-blocks", "1",
                "--n-heads", "4",
                "--minimum-recall", str(args.minimum_recall),
                "--device", args.device,
            ],
            env,
            args.dry_run,
        )

    print("outputs:", flush=True)
    print(f"  dataset: {dataset_dir}", flush=True)
    print(f"  cache: {cache}", flush=True)
    print(f"  temporal head: {temporal_head}", flush=True)
    if args.head == "both":
        print(f"  conv-attention head: {attention_head}", flush=True)


if __name__ == "__main__":
    main()
