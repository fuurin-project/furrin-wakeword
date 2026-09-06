#!/usr/bin/env python3
from __future__ import annotations

import argparse
import queue
import time
from pathlib import Path

import numpy as np
import soundfile as sf


SAMPLE_RATE = 16_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the distilled Speech Commands backbone from a microphone or WAV."
    )
    parser.add_argument("wav", type=Path, nargs="?", help="Optional mono WAV instead of mic")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("runs/student_balanced_57078/student_best.pt"),
    )
    parser.add_argument(
        "--constants", type=Path, default=Path("artifacts/fuurin_frontend.npz")
    )
    parser.add_argument(
        "--head", type=Path, default=Path("artifacts/speech_commands_student_head.npz")
    )
    parser.add_argument("--device", help="Microphone device index or name")
    parser.add_argument("--compute-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--loop", action="store_true", help="Press Enter for each recording")
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Continuously classify an 80-ms-updated rolling two-second window",
    )
    parser.add_argument(
        "--min-rms",
        type=float,
        default=100.0,
        help="Mark quieter windows as silence instead of a trusted prediction",
    )
    parser.add_argument(
        "--save", type=Path, default=Path("recordings/latest_command.wav")
    )
    return parser.parse_args()


def sounddevice_module():
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit("Microphone mode requires sounddevice") from exc
    return sd


def record(duration: float, device: str | None) -> np.ndarray:
    if duration <= 0:
        raise SystemExit("--duration must be positive")
    sd = sounddevice_module()
    selected: int | str | None = int(device) if device and device.isdigit() else device
    print(f"Recording {duration:.1f}s at 16 kHz -- speak one command now...", flush=True)
    try:
        audio = sd.rec(
            int(round(duration * SAMPLE_RATE)), samplerate=SAMPLE_RATE,
            channels=1, dtype="int16", device=selected,
        )
        sd.wait()
    except Exception as exc:
        raise SystemExit(
            f"Microphone recording failed: {exc}\nRun with --list-devices and pass --device ID."
        ) from exc
    return np.asarray(audio, dtype=np.int16).reshape(-1)


def read_wav(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="int16", always_2d=False)
    if sample_rate != SAMPLE_RATE:
        raise SystemExit(f"WAV must be 16 kHz, got {sample_rate} Hz")
    if audio.ndim == 2:
        audio = np.rint(audio.astype(np.float32).mean(axis=1)).astype(np.int16)
    return np.asarray(audio, dtype=np.int16)


def load_models(args: argparse.Namespace):
    from oww_distill.classifier import DistilledStudentRuntime, LinearCommandClassifier

    runtime = DistilledStudentRuntime(args.checkpoint, args.constants, args.compute_device)
    classifier = LinearCommandClassifier.load(args.head)
    return runtime, classifier


def predict(pcm: np.ndarray, runtime, classifier) -> tuple[np.ndarray, float, float]:
    started = time.perf_counter()
    probabilities = classifier.probabilities(runtime.embed_pcm(pcm))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))
    return probabilities, rms, elapsed_ms


def infer(pcm: np.ndarray, args: argparse.Namespace) -> None:
    # Import torch-backed runtime only after microphone capture. This keeps
    # PortAudio device enumeration independent of ML runtime initialization.
    runtime, classifier = load_models(args)
    probabilities, rms, elapsed_ms = predict(pcm, runtime, classifier)
    top_k = min(max(args.top_k, 1), len(probabilities))
    indices = np.argsort(probabilities)[-top_k:][::-1]
    prediction = classifier.classes[indices[0]] if rms >= args.min_rms else "[silence]"
    print(f"prediction: {prediction}")
    print(f"audio_rms: {rms:.1f}  inference: {elapsed_ms:.2f} ms")
    for index in indices:
        print(f"  {classifier.classes[index]:>10s}  {probabilities[index]:7.2%}")


def realtime(args: argparse.Namespace) -> None:
    sd = sounddevice_module()
    selected: int | str | None = int(args.device) if args.device and args.device.isdigit() else args.device
    runtime, classifier = load_models(args)
    audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=8)
    block_samples = 1_280  # openWakeWord-compatible 80-ms cadence
    window_samples = 32_000
    rolling = np.zeros(window_samples, dtype=np.int16)
    received = 0

    def callback(indata, frames, time_info, status) -> None:
        del frames, time_info
        if status:
            print(f"\naudio status: {status}", flush=True)
        chunk = np.frombuffer(indata, dtype=np.int16).copy()
        try:
            audio_queue.put_nowait(chunk)
        except queue.Full:
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                pass
            audio_queue.put_nowait(chunk)

    print("Realtime mode: speak one of the 35 commands; press Ctrl+C to stop.")
    print("The rolling window warms up for 2 seconds.")
    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=block_samples,
            channels=1,
            dtype="int16",
            device=selected,
            callback=callback,
        ):
            while True:
                chunk = audio_queue.get()
                if len(chunk) != block_samples:
                    continue
                rolling[:-block_samples] = rolling[block_samples:]
                rolling[-block_samples:] = chunk
                received += block_samples
                if received < window_samples:
                    print(
                        f"\rwarming up: {100.0 * received / window_samples:5.1f}%",
                        end="", flush=True,
                    )
                    continue
                probabilities, rms, elapsed_ms = predict(rolling, runtime, classifier)
                indices = np.argsort(probabilities)[-3:][::-1]
                if rms < args.min_rms:
                    summary = "[silence]"
                else:
                    summary = "  ".join(
                        f"{classifier.classes[index]} {probabilities[index]:.1%}"
                        for index in indices
                    )
                print(
                    f"\rRMS {rms:7.1f} | {elapsed_ms:6.2f} ms | {summary:<55}",
                    end="", flush=True,
                )
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as exc:
        raise SystemExit(
            f"Realtime microphone failed: {exc}\n"
            "Run with --list-devices and pass --device ID."
        ) from exc


def capture_and_run(args: argparse.Namespace) -> None:
    pcm = record(args.duration, args.device)
    args.save.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.save, pcm, SAMPLE_RATE, subtype="PCM_16")
    print(f"saved: {args.save}")
    infer(pcm, args)


def main() -> None:
    args = parse_args()
    if args.list_devices:
        print(sounddevice_module().query_devices())
        return
    if args.wav is not None:
        if args.realtime or args.loop:
            raise SystemExit("WAV input cannot be combined with --realtime or --loop")
        infer(read_wav(args.wav), args)
        return
    if args.realtime:
        realtime(args)
        return
    if not args.loop:
        capture_and_run(args)
        return
    while True:
        answer = input("Press Enter to record, or q then Enter to quit: ").strip().lower()
        if answer == "q":
            return
        capture_and_run(args)


if __name__ == "__main__":
    main()
