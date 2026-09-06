# openWakeWord Speech Embedding Distillation Design

Status: approved for host implementation on 2026-09-02.

## Current v0 distillation result

The first host baseline completed on an RTX 5070 Ti using 57,078 balanced
clips: 28,539 LibriSpeech train-clean-100 clips and 28,539 sampled Speech
Commands v0.02 clips. The deterministic group split contains 44,952 training,
6,425 validation, and 5,701 held-out test clips.

The 64-channel causal student has 34,848 deployable parameters and produces
`16 x 96` outputs from `197 x 32` input features. After 30 epochs:

- best validation centered cosine: 0.689718;
- held-out test centered cosine: 0.685978;
- held-out test raw cosine: 0.907009;
- held-out test normalized MSE: 0.582322;
- training time: 36.58 seconds;
- input quantizer clipped fraction: 0.000499;
- estimated INT8 weight storage: 34,848 bytes before metadata.

The raw-cosine constant-mean baseline is already 0.825927, so centered cosine
is the meaningful distillation diagnostic. This checkpoint is still a float
PyTorch training artifact with an INT8-simulated input boundary; it is not yet
a fully quantized deployable model. Downstream `Hey Pico` transfer remains the
next quality gate.

## Generic v1 temporal-distillation result

The unchanged 64-channel, 34,848-parameter student was retrained for 150 epochs
with cosine learning-rate decay and an auxiliary MSE loss over adjacent
80-ms teacher-embedding deltas. It remains keyword-independent: no `Hey Pico`
labels or wake-word classification loss were used.

- best validation centered cosine: 0.740500;
- held-out test centered cosine: 0.736651;
- held-out test raw cosine: 0.921562;
- held-out test normalized MSE: 0.494392;
- training time: 198.80 seconds on RTX 5070 Ti;
- Speech Commands frozen-probe top-1 accuracy: 85.43%.

Relative to v0, held-out centered cosine improves by 0.050673 and the matched
Speech Commands probe improves by 3.32 percentage points. The teacher remains
at 91.55%, leaving a 6.11-point downstream gap. Artifacts are in
`runs/student_generic_v1_delta/` and
`runs/speech_commands_linear_probe_generic_v1_delta/`.

## Frozen Speech Commands linear probe

A matched downstream probe was completed on the Speech Commands portion of the
formal cache. It contains 35 commands with 22,854 training, 3,035 validation,
and 2,650 held-out test clips. The existing source-speaker hash split was kept
unchanged. Each backbone used temporal mean pooling to 96 dimensions followed
by an independently fitted but identically configured standard scaler and
multiclass logistic-regression head (3,395 parameters, seed 17, fixed C=1.0).

Held-out test results:

| Frozen backbone | Accuracy | Balanced accuracy | Macro F1 | Top-5 accuracy |
| --- | ---: | ---: | ---: | ---: |
| openWakeWord teacher | 91.55% | 90.82% | 91.00% | 97.74% |
| distilled student | 82.11% | 80.69% | 80.83% | 95.96% |
| untrained random student | 33.47% | 31.70% | 31.54% | 70.26% |

The distilled model gains 48.64 top-1 percentage points over the random
student and trails the teacher by 9.43 points. This demonstrates retained
generic keyword information under a frozen linear probe, but it does not prove
`Hey Pico` recall/FAR, continuous false activations, or RP2350 performance.
The machine-readable report is
`runs/speech_commands_linear_probe/metrics.json`.

## Objective and priority

Distill openWakeWord's shared speech embedding encoder into an RP2350-sized
student. The priorities, in order, are:

1. deployability on RP2350 under Zephyr;
2. transfer to a newly trained wake-word head;
3. numerical similarity to the teacher embedding.

The first wake phrase is US-English `Hey Pico`. Positive and phonetic
hard-negative speech is generated only with Piper. Public recorded datasets
may be used for general-speech and background negatives. No real-speaker
positive evaluation is available, so v0 results are synthetic/offline evidence
and must not be reported as real-speaker or deployment performance.

## Compatibility boundary

The wake-word head may be retrained. The student retains the useful upstream
contract of one 96-dimensional embedding per 80 ms update, but it is not
required to reproduce teacher values bit-for-bit or run an existing head
unchanged.

The frontend signal definition is kept compatible with openWakeWord. Only its
implementation changes from a large float TFLite DFT graph to CMSIS-DSP.

## Reference frontend

Input is 16-kHz mono PCM. One feature frame uses:

- a 400-sample (25 ms) Hann window;
- zero padding to a 512-point real FFT;
- a 160-sample (10 ms) hop;
- 257 power-spectrum bins;
- the original sparse 257-by-32 Mel filterbank;
- the original 10*log10 compression and 80 dB dynamic-range floor;
- the openWakeWord output transform `feature / 10 + 2`.

The original local `melspectrogram.tflite` is 1,092,516 bytes. Its two
float32 DFT convolution constants consume 526,336 bytes each. The Mel matrix
has only 229 non-zero coefficients. The host DSP reference replaces the DFT
convolutions with an RFFT but keeps the window, Mel weights, compression, and
frame alignment.

Development proceeds in two steps:

1. float32 host DSP implementation and parity against the TFLite frontend;
2. Zephyr CMSIS-DSP float32 implementation using the same exported constants.

Frontend output is quantized only at the student boundary. Quantization uses
frozen affine parameters derived from the training split. Validation, test,
host inference, and firmware must use the same scale and zero point.

## Teacher and student data flow

For each identical two-second PCM window:

```text
PCM -> original openWakeWord frontend + encoder -> 16 x 96 teacher targets
 |
 +-> DSP-equivalent frontend -> affine int8 features -> tiny student
```

The teacher runs only during host-side cache generation. It is never included
in firmware.

The first student is a causal depthwise-separable temporal convolutional
network. It consumes 32-bin frames, preserves streaming state, and produces a
96-dimensional output every eight new 10-ms frames. Training may use a smaller
internal embedding followed by a 96-dimensional compatibility projection.
Candidate model widths are evaluated by deployable parameter count and
downstream head metrics rather than teacher cosine alone.

## Training objectives

The backbone loss combines:

- standardized teacher regression/cosine loss;
- keyword/command classification on generic speech where labels exist;
- supervised contrastive loss across keyword identities and augmentations.

Teacher targets are centered and standardized per dimension using the training
split. Raw cosine is not a valid primary metric because openWakeWord embeddings
contain a strong common direction; a constant mean vector can otherwise score
misleadingly well.

`Hey Pico` is excluded from generic backbone pretraining. It is used only after
the shared student is frozen to measure new-head transfer. If the frozen
student cannot separate positives from speech negatives, end-to-end fine-tuning
may be evaluated as a separate model and clearly labeled.

## Data and splits

Backbone data:

- LibriSpeech train-clean-100;
- Speech Commands v0.02;
- silence and background noise bundled with Speech Commands.

Wake-word data:

- Piper US-English `Hey Pico` positives;
- Piper phonetic and lexical hard negatives such as `Hey Peter`, `Hey people`,
  `Okay Pico`, and `Hey pickle`;
- LibriSpeech and Speech Commands general-speech negatives.

Piper voice IDs are disjoint across train, validation, and test. The initial
target is eight training voices, two validation voices, and two test voices.
All augmentations derived from one source clip remain in the same split.

v0 intentionally excludes room impulse response, far-field microphone,
TV/music stress, and SNR sweeps. Those are v1 robustness work, not v0 gates.

## Wake-word head and metrics

The head is retrained from student embeddings. Model selection is false-trigger
first subject to a minimum held-out Piper recall of 90%.

v0 reports:

- recall on held-out Piper voices;
- false-accept rate on Piper hard negatives;
- false activations on clean continuous general-speech negatives;
- ROC and recall-constrained operating points;
- teacher versus student head results under matched splits and update rules;
- frontend parity, model size, and host latency.

The provisional clean baseline targets are recall at least 90%, hard-negative
false accepts at most 1%, and as few general-speech false triggers as possible.
These do not establish real-speaker recall or real-world false accepts per hour.

### Generic-v1 streaming-aligned head comparison

The frozen generic-v1 student was evaluated with both the original lightweight
temporal head and a LiveKit-inspired Conv-Attention head. To model the rolling
detector without leaking labels through position, every speech class (positive,
phonetic negative, and general speech) was aligned near the trailing edge with
the same deterministic 0--200 ms jitter. Pure noise remained position-free.

| Frozen-student head | Parameters | Validation recall | Validation FAR | Test recall | Test FAR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Temporal max/mean | 1,969 | 90.28% | 1.250% | 97.22% | 0.764% |
| Conv-Attention small | 18,721 | 90.28% | 1.384% | 99.07% | 0.859% |

The Conv-Attention head improves held-out recall but does not beat the smaller
head's clip FAR at the validation-selected operating point. Its remaining test
false accepts are limited to synthetic Piper hard negatives. Checkpoint
averaging was evaluated, but the best single checkpoint won on validation FAR.
Neither result establishes continuous false activations per hour.

## Host-first deliverable

The independent host project must provide:

- extraction of the original frontend constants;
- DSP frontend and golden-vector parity tests;
- deterministic dataset manifests and speaker/voice-disjoint splits;
- teacher cache generation with resumable output;
- student training and checkpoint export;
- wake-word head training and calibration;
- WAV batch evaluation and microphone streaming inference;
- machine-readable metrics including every unproven acceptance gate.

## Zephyr migration

After the host baseline is validated, create a standalone Zephyr application
for `rpi_pico2/rp2350a/m33/w`:

```text
audio/int16 ring buffer
  -> CMSIS-DSP float32 frontend
  -> frozen affine int8 quantizer
  -> fully-int8 student
  -> retrained int8 wake-word head
  -> temporal decision and cooldown
```

The initial cadence matches openWakeWord: 1,280 new samples per update, eight
new Mel frames, and one 96-dimensional embedding every 80 ms. Each complete
update must finish before the next 80-ms chunk. Final flash, RAM, stack, tensor
arena, and latency limits are set from measured Zephyr baselines rather than
claimed from host timings.

## Validation gates

1. DSP and TFLite frontend outputs have the same frame count and alignment.
2. Frontend error is reported on silence, impulse, tones, noise, and speech.
3. Student beats the constant-mean teacher-target baseline.
4. Teacher and student heads use identical positive/negative splits.
5. Held-out Piper recall is at least 90% before false-trigger comparison.
6. Artifacts contain frontend constants, quantization parameters, architecture,
   split hashes, seed, and model/head pairing metadata.
7. Zephyr build success is not treated as device performance; timing and memory
   must be measured on RP2350.
