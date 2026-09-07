# Fuurin wake-word distillation

Host-first distillation of a reusable 96-D speech embedder for wake-word heads
and later Zephyr/RP2350 deployment. The shared embedder is independent of any
specific wake phrase; `Hey Pico`, `Hey Nordic`, and future phrases use separate
small heads.

See [`MODEL_RESULTS.md`](MODEL_RESULTS.md) for the selected model metrics and
SHA-256 values, and
[`docs/openwakeword_embedding_distillation_design.md`](docs/openwakeword_embedding_distillation_design.md)
for the frontend and model design.

## Repository layout

```text
artifacts/              Canonical host and MCU runtime files tracked by Git
src/oww_distill/        Frontend, embedder, classifiers, and shared utilities
scripts/                Dataset, training, evaluation, and live-test commands
tests/                  Automated runtime and model-shape checks
docs/                   Design documentation
data/                   Generated audio and caches; ignored by Git
runs/                   Temporary checkpoints and full training logs; ignored by Git
```

`runs/` is disposable training state. A training command writes candidate
checkpoints and detailed epoch history there. Once a model is selected, only
the deployable artifact is promoted to `artifacts/`; final evidence is copied
into `MODEL_RESULTS.md`.

## Canonical runtime artifacts

```text
artifacts/
├── fuurin_frontend.npz  Shared PCM-to-log-Mel constants
├── fuurin_embed.pt      Shared distilled speech embedder
├── fuurin_embed.tflite  Full-INT8 one-frame streaming student
├── hey_pico_head.pt     Selected Hey Pico Conv-Attention head
├── hey_pico_head.tflite Full-INT8 Hey Pico Conv-Attention head
└── hey_nordic_head.npz  Selected Hey Nordic temporal head
```

Git commits identify artifact versions, so filenames remain stable.

## Setup and tests

```bash
uv sync --extra test
uv run pytest -q
```

Piper TTS generation additionally needs the data extra:

```bash
uv sync --extra data --extra test
```

## Live testing

The live runtime consumes 1,280-sample chunks, updates every 80 ms, and scores
a rolling two-second 16-kHz mono window.

List microphone devices:

```bash
PYTHONPATH=src .venv/bin/python scripts/live_hey_pico.py --list-devices
```

`Hey Pico` is the default canonical pair:

```bash
PYTHONPATH=src .venv/bin/python scripts/live_hey_pico.py --device DEVICE
```

Test `Hey Nordic` with the same frontend and embedder:

```bash
PYTHONPATH=src .venv/bin/python scripts/live_hey_pico.py --device DEVICE \
  --head artifacts/hey_nordic_head.npz
```

For a WAV smoke test, pass a 16-kHz WAV as the positional argument.

## Train another wake phrase

The generic entry point creates a speaker-disjoint Piper dataset, adds open
speech/noise negatives, applies the same trailing-edge plus 0--200 ms jitter to
all speech, builds an embedding cache, and trains a phrase-specific head.

The default is the 1,969-parameter temporal head:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_wake_phrase.py \
  --wake-phrase "Hey Nordic" \
  --hard-negative-text "Hey Nautic" \
  --hard-negative-text "Hey Norbit"
```

Use `--head both` to also train the larger Conv-Attention comparison. The two
component-only phrases are always included as weighted negatives. Repeated
`--hard-negative-text` values supply the remaining phonetic confusers.

Useful controls:

```text
--dry-run          Print the complete pipeline without training
--force-dataset    Regenerate Piper audio
--rebuild-cache    Rebuild embeddings after audio or backbone changes
--piper-python     Select a Python environment containing piper-tts
--device           auto, cpu, or cuda
```

Generated datasets go under `data/`, full metrics and temporary checkpoints go
under `runs/`, and candidate heads go under `artifacts/` but remain ignored
until explicitly promoted.

## Rebuild the shared embedder

```bash
PYTHONPATH=src .venv/bin/python scripts/export_frontend.py \
  --model /path/to/melspectrogram.tflite

PYTHONPATH=src .venv/bin/python scripts/build_teacher_cache.py \
  --librispeech /path/to/LibriSpeech/train-clean-100 \
  --speech-commands /path/to/speech_commands_v0.02

PYTHONPATH=src .venv/bin/python scripts/train_student.py \
  --cache data/cache/distill_balanced_57078.npz \
  --run-dir runs/student_candidate
```

Promoting a new embedder requires matched downstream probes and regeneration
of any paired head caches.

## Export the streaming INT8 student

Install the optional TensorFlow exporter and quantize with streaming-aligned
wake-word audio. The model consumes one quantized 32-bin log-Mel frame per call;
firmware calls it eight times for each 80-ms audio update and carries its six
explicit state tensors forward.

```bash
uv sync --extra test --extra export

PYTHONPATH=src .venv/bin/python scripts/export_tflite_student.py \
  --metadata-cache \
    data/cache/hey_pico_piper_open_speech_student_generic_v1_delta_streaming.npz \
  --calibration-clips 256 \
  --validation-clips 32 \
  --calibration-stride 4
```

The command writes Float32 and full-INT8 candidates plus parity metrics under
`runs/tflite_student/`. Promote the verified INT8 file as
`artifacts/fuurin_embed.tflite`, then run the downstream test:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_tflite_student.py
```

The frozen feature normalization is folded into the first convolution, so the
MCU feeds frontend log-Mel values directly. The student TFLite signature
contains `feature`, `state_0` ... `state_5` inputs
and `embedding`, `next_state_0` ... `next_state_5` outputs. State output scales
match their corresponding next-call input scales, so firmware can retain the
INT8 buffers without requantization.

Export the selected Conv-Attention head with full-INT8 input/output and validate
the complete student/head pair:

```bash
PYTHONPATH=src .venv/bin/python scripts/export_tflite_head.py

PYTHONPATH=src .venv/bin/python scripts/validate_tflite_student.py \
  --tflite-head artifacts/hey_pico_head.tflite \
  --output runs/tflite_head/full_int8_hey_pico.json
```

The student embedding output uses `(scale=0.0303696655, zero_point=5)` while
the head input uses `(scale=0.0383578427, zero_point=-1)`. Firmware must
requantize the 16x96 embedding ring between the two models. The head requires
TFLite operators including `BATCH_MATMUL`, `SOFTMAX`, `RSQRT`, and
`SQUARED_DIFFERENCE`; TFLite Micro kernel availability, tensor-arena size, and
RP2350 latency must be validated in the target build.

Host realtime testing of both INT8 models is available with:

```bash
PYTHONPATH=src .venv/bin/python -u scripts/live_hey_pico.py \
  --device DEVICE \
  --tflite-model artifacts/fuurin_embed.tflite \
  --tflite-head artifacts/hey_pico_head.tflite
```

## Evidence boundary

Current reported results are voice-disjoint synthetic/offline clip metrics.
They do not establish real-speaker recall, continuous false accepts per hour,
or RP2350 timing/RAM/flash performance. The student and Hey Pico head have
validated full-INT8 TFLite exports, but the frontend remains Float32 DSP and
target TFLite Micro operator coverage, requantization, tensor arena, and
on-device timing remain separate MCU acceptance steps.
