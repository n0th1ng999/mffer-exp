# 0016. Every run is deterministic: a rerun on the same machine reproduces it bit for bit

- **Status:** Accepted, 20 Nov 2026. Supersedes the "CUDA is not bit-for-bit deterministic" consequence of
  [0015](0015-cuda-allowed-metal-not.md) and the README's "a rerun can differ by a few tenths of a point"
- **Where:** `mffer/device.py` (`configure` enables op determinism, `deterministic`), `mffer/models.py`
  (`derive_seed`, `reseed`), `mffer/train.py` (`Reseed` callback, `first_epoch`, `meta.json`
  `deterministic`), `mffer/data/pipeline.py` (`make_dataset(first_epoch=)`), `mffer/federated.py`
  (per-round `reseed`, results sorted by client before FedAvg), `scripts/lab_stability.py` (uses
  `mffer.device`), `tests/conftest.py`, `tests/test_determinism.py`, `tests/test_pipeline.py`

## Context

The repository exists so that every number in the paper can be traced to a run and rerun. Seeds were
already fixed (weights: split id; shuffles: seed + epoch; federated dropout: seed, client, round), but four
things still let two runs of the same command differ:

1. TensorFlow's default kernels are not deterministic: cuDNN picks algorithms and GPU reductions sum in a
   varying order. On the RTX 4080 two identical 20-step trainings already differed by 2e-7, and
   `tests/test_federated.py` (FedProx with μ = 0 equals plain training) failed on CUDA only.
2. Each Keras `Dropout` layer owns a seed generator whose state advances with every step and is rebuilt
   when a model is loaded. A resumed run (`--resume`) therefore drew other dropout masks than the same run
   uninterrupted, and a federated client's masks depended on which Ray worker process had run before.
3. `make_dataset` counted shuffle epochs from 0 each time it was built, so a run resumed after epoch *k*
   replayed epoch 0's order instead of epoch *k*'s.
4. Flower hands `aggregate_fit` the client results in arrival order; FedAvg's float sums then depend on
   which client finished first.

## Decision

- `mffer.device.configure()`, which every entry point, every Flower client process and the test session
  call before TensorFlow runs an op, calls `tf.config.experimental.enable_op_determinism()` on every
  platform and device. An op without a deterministic kernel raises instead of running.
- Every derived seed comes from one rule, `models.derive_seed(*ints)` (NumPy `SeedSequence`).
- Dropout state is set explicitly, never inherited: `models.reseed(model, seed)` sets each random layer's
  state from the seed. Centralised and local-only training reseed at the start of every epoch from
  (run seed, epoch); a federated client reseeds every round from (run seed, client, round).
- A resumed run shuffles from the epoch it resumes at (`make_dataset(first_epoch=len(history))`).
- `aggregate_fit` sorts client results by client id before averaging.
- `meta.json` records `"deterministic": true`.

## Verification (20 Nov 2026, RTX 4080 / Ryzen 9 7950X, WSL2, TF 2.19.1, Keras 3.15.1)

- All eight presets (`m1`, `d1`–`d4`, `m3`, `audio_only`, `video_only`) have deterministic CUDA kernels and
  train to bit-identical weights from the same seed.
- End to end, a run interrupted halfway and resumed (`--resume`) matches the same run uninterrupted bit for
  bit: every `epochs.csv` value (max difference 0.0), every array of `best.weights.h5` and `test.json`:

  | Run (split 0) | Device | Epochs / rounds | Resumed after | Identical |
  |---|---|---|---|---|
  | `mffer.train`, audio_only | CUDA | 4 | 2 | yes |
  | `mffer.train`, m3 | CUDA | 3 | 2 | yes |
  | `mffer.federated`, 2 actor clients, audio_only | CUDA (clients on CUDA via Ray) | 3 | 2 | yes |
  | `mffer.train`, audio_only | CPU | 4 | 2 | yes |

  Control: the same audio_only CUDA check on the code before this record gives different weights and test
  metrics (epoch metrics up to 0.025 apart).
- Cost: none measurable. M3 0.156 s/step on CUDA with determinism vs 0.157 without (smoke test).

## Consequences

- Same code (`git describe`), data, seed, device and software stack → identical run, including after a
  resume and in federated runs. Tests check the mechanisms (`tests/test_determinism.py`, the resumed-shuffle
  test in `tests/test_pipeline.py`, and the FedProx μ = 0 equality now holding on CUDA too).
- Determinism is per platform, not across platforms: CPU vs CUDA, two CPU models (oneDNN picks
  ISA-specific kernels) or two TensorFlow versions still differ by float rounding, which over a run grows
  into differences of a few tenths of a point (0015's check). Runs from different devices are compared as
  mean ± sd over the three splits, as before; the ledger's Device column says which is which.
- Runs started before this record were not deterministic; they stay valid under 0015 but cannot be
  reproduced bit for bit. A paper run should be started from a committed tree (`git describe` without
  `-dirty`) so that its code can be checked out again.
- A future op without a deterministic kernel fails loudly; the fix is another layer or op, not turning
  determinism off.
- Bug fix, 24 Nov 2026: the training shuffle still used `seed + epoch`, so seed *s*'s epoch *e*+1 repeated
  seed *s*+1's epoch *e* (a `--seed` variance check on one split reused one shuffle sequence, shifted by one
  epoch). It now follows the rule above: `derive_seed(seed, epoch)` (`pipeline.make_dataset`). Batch order
  of every centralised run changes; runs from before the fix are protocol 1 (see 0017).
