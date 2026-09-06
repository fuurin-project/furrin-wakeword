#!/usr/bin/env python3
from __future__ import annotations

import argparse
import queue
import time
from pathlib import Path

import numpy as np
import soundfile as sf


SAMPLE_RATE = 16_000
CHUNK_SAMPLES = 1_280
WINDOW_SAMPLES = 32_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime test for a paired 34k distilled backbone and wake-word head."
    )
    parser.add_argument("wav", type=Path, nargs="?", help="Optional 16-kHz WAV smoke test")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("artifacts/fuurin_embed.pt"),
    )
    parser.add_argument(
        "--constants", type=Path, default=Path("artifacts/fuurin_frontend.npz")
    )
    parser.add_argument(
        "--head", type=Path,
        default=Path("artifacts/hey_pico_head.pt"),
    )
    parser.add_argument("--device", help="Microphone input device index or name")
    parser.add_argument("--compute-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threshold", type=float, help="Override calibrated head threshold")
    parser.add_argument("--min-rms", type=float, default=100.0)
    parser.add_argument("--cooldown", type=float, default=1.5)
    parser.add_argument("--list-devices", action="store_true")
    return parser.parse_args()


def sounddevice_module():
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit("Microphone mode requires sounddevice") from exc
    return sd


def load_models(args: argparse.Namespace):
    from oww_distill.classifier import DistilledStudentRuntime, load_wake_classifier

    runtime = DistilledStudentRuntime(args.checkpoint, args.constants, args.compute_device)
    detector = load_wake_classifier(args.head, args.compute_device)
    threshold = detector.threshold if args.threshold is None else args.threshold
    if not 0 <= threshold <= 1:
        raise SystemExit("--threshold must be between zero and one")
    return runtime, detector, threshold


def read_wav(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="int16", always_2d=False)
    if sample_rate != SAMPLE_RATE:
        raise SystemExit(f"WAV must be 16 kHz, got {sample_rate}")
    if audio.ndim == 2:
        audio = np.rint(audio.astype(np.float32).mean(axis=1)).astype(np.int16)
    return np.asarray(audio, dtype=np.int16)


def end_pad_or_trim(pcm: np.ndarray, samples: int = WINDOW_SAMPLES) -> np.ndarray:
    """Place an isolated wake-word WAV at the live rolling window's trailing edge."""
    pcm = np.asarray(pcm, dtype=np.int16).reshape(-1)
    if len(pcm) >= samples:
        return pcm[-samples:].copy()
    output = np.zeros(samples, dtype=np.int16)
    output[-len(pcm) :] = pcm
    return output


def score(pcm: np.ndarray, runtime, detector) -> tuple[float, float, float]:
    started = time.perf_counter()
    value = detector.score(runtime.embed_frames_pcm(pcm))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))
    return value, rms, elapsed_ms


def wav_mode(path: Path, args: argparse.Namespace) -> None:
    runtime, detector, threshold = load_models(args)
    value, rms, elapsed_ms = score(end_pad_or_trim(read_wav(path)), runtime, detector)
    detected = rms >= args.min_rms and value >= threshold
    print(f"{detector.name}_score: {value:.6f}")
    print(f"threshold:      {threshold:.6f}")
    print(f"audio_rms:      {rms:.1f}")
    print(f"inference_ms:   {elapsed_ms:.2f}")
    print(f"detected:       {detected}")


def realtime(args: argparse.Namespace) -> None:
    sd = sounddevice_module()
    selected: int | str | None = int(args.device) if args.device and args.device.isdigit() else args.device
    runtime, detector, threshold = load_models(args)
    display_name = detector.name.replace("_", " ").upper()
    chunks: queue.Queue[np.ndarray] = queue.Queue(maxsize=8)
    rolling = np.zeros(WINDOW_SAMPLES, dtype=np.int16)
    received = 0
    last_trigger = -float("inf")

    def callback(indata, frames, time_info, status) -> None:
        del frames, time_info
        if status:
            print(f"\naudio status: {status}", flush=True)
        chunk = np.frombuffer(indata, dtype=np.int16).copy()
        try:
            chunks.put_nowait(chunk)
        except queue.Full:
            try:
                chunks.get_nowait()
            except queue.Empty:
                pass
            chunks.put_nowait(chunk)

    print(
        f"Realtime {display_name}: threshold={threshold:.3f}, update=80 ms, "
        "window=2 s. Press Ctrl+C to stop."
    )
    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE, blocksize=CHUNK_SAMPLES, channels=1,
            dtype="int16", device=selected, callback=callback,
        ):
            while True:
                chunk = chunks.get()
                if len(chunk) != CHUNK_SAMPLES:
                    continue
                rolling[:-CHUNK_SAMPLES] = rolling[CHUNK_SAMPLES:]
                rolling[-CHUNK_SAMPLES:] = chunk
                received += CHUNK_SAMPLES
                if received < WINDOW_SAMPLES:
                    print(
                        f"\rwarming up: {100 * received / WINDOW_SAMPLES:5.1f}%",
                        end="", flush=True,
                    )
                    continue
                value, rms, elapsed_ms = score(rolling, runtime, detector)
                now = time.monotonic()
                detected = (
                    rms >= args.min_rms and value >= threshold
                    and now - last_trigger >= args.cooldown
                )
                if detected:
                    last_trigger = now
                    print(
                        f"\n>>> {display_name} DETECTED score={value:.3f} "
                        f"rms={rms:.0f} inference={elapsed_ms:.2f} ms <<<",
                        flush=True,
                    )
                else:
                    bar_count = int(round(value * 30))
                    bar = "#" * bar_count + "." * (30 - bar_count)
                    print(
                        f"\rscore {value:6.3f} [{bar}] | RMS {rms:7.1f} | "
                        f"{elapsed_ms:6.2f} ms",
                        end="", flush=True,
                    )
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as exc:
        raise SystemExit(
            f"Realtime microphone failed: {exc}\n"
            "Run --list-devices and select an input with --device ID."
        ) from exc


def main() -> None:
    args = parse_args()
    if args.cooldown < 0 or args.min_rms < 0:
        raise SystemExit("cooldown and minimum RMS cannot be negative")
    if args.list_devices:
        print(sounddevice_module().query_devices())
        return
    if args.wav is not None:
        wav_mode(args.wav, args)
        return
    realtime(args)


if __name__ == "__main__":
    main()
