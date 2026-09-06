# Selected model results

This file records compact evidence for the canonical runtime artifacts. Git
history identifies artifact versions; SHA-256 verifies deployed copies.

## Shared frontend and embedder

| Artifact | Size | SHA-256 |
|---|---:|---|
| `artifacts/fuurin_frontend.npz` | 3,746 B | `fc1d3f2dd20662f58950222ef031faa2c5960bbb9d0df0785ae502975dfdc50d` |
| `artifacts/fuurin_embed.pt` | 152,449 B | `ed1f6cc1de8cb59d28a2ff795da470470fb24c91857b270b958feca96a49766c` |

`fuurin_embed.pt` is a 34,848-parameter causal depthwise-separable student. It
maps `(197, 32)` log-Mel input to `(16, 96)` embeddings and does not use wake
phrase labels. The selected temporal-delta run reached 0.736651 held-out
centered cosine and 85.43% held-out Speech Commands top-1 accuracy under the
matched frozen probe.

## Hey Pico

| Field | Value |
|---|---|
| Artifact | `artifacts/hey_pico_head.pt` |
| SHA-256 | `40654dad8152e2c6195b6856658a0b81d870d252a646b33224c4360e84602219` |
| Size | 83,525 B |
| Architecture | Conv-Attention, 18,721 parameters |
| Threshold | 0.288866 |
| Test recall | 99.07% |
| Test clip FAR | 0.859% |

Evaluation used the voice-disjoint, streaming-aligned Piper/open-speech split.

## Hey Nordic

| Field | Value |
|---|---|
| Artifact | `artifacts/hey_nordic_head.npz` |
| SHA-256 | `77382b95564e8f0fb8b2d7a92a7a5e880f17de97bd55052b5b9a3a5f65dccfa0` |
| Size | 10,718 B |
| Architecture | Temporal depthwise/pointwise head, 1,969 parameters |
| Threshold | 0.501859 |
| Test recall | 98.15% |
| Test clip FAR | 0.525% |
| Component-only FAR | 0% for `Hey`; 0% for `Nordic` |

The full training set contained 19,963 clips, including 1,800 Piper positives,
300 component-only negatives, 2,700 Piper hard negatives, and 15,163
open-speech/noise negatives. The speaker-disjoint test split contained 216
positives and 2,095 negatives.

## Limitations

These are offline clip results from synthetic, open-speech, and noise data.
They are not continuous-FAPH, real-speaker, or RP2350 deployment evidence.
