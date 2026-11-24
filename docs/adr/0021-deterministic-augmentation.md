# 0021. Deterministic training-time augmentation

- **Status:** Proposed, 24 Nov 2026 (flag `augment: true`; decided by the dev sweep, 0017)
- **Where:** `mffer/data/pipeline.py` (`augment`, `make_dataset`), `tests/test_protocol2.py`

## Context

The training split has 63 actors (about 5,100 clips). Models trained from scratch on so few people
overfit to identity. Augmentation must keep 0016's bit-for-bit reproducibility, including resume and FL.

## Decision

With `augment: true`, training batches only (never validation or test):
- video: per clip, one horizontal flip (p 0.5) and one random square crop of 200–224 px resized back to
  224 (cv2 INTER_AREA), the same for all 10 frames;
- log-mel: per clip, 2 time masks (each up to 10% of the real frames) and 2 frequency masks (each up to
  8 bands), set to 0 (the normalised mean). MFCCs are not masked.
Draws come from NumPy `default_rng(derive_seed(seed, epoch, 1))` in a fixed order, so a seed and epoch
reproduce them, and a resumed run or a federated client (seed from client and round) draws the same.

## Consequences

- Reproducibility kept (tested; the crash/resume check is bit-identical with augmentation on).
- Augmentation runs in the input generator on the CPU; the timing pilot reports its cost.
