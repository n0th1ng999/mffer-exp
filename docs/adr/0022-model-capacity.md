# 0022. Model capacity is chosen on validation

- **Status:** Proposed, 24 Nov 2026 (flags `width: 1 | 2`, `visual_blocks: 3 | 4`; decided by the dev
  sweep, 0017)
- **Where:** `mffer/models.py` (`ModelSpec.width`, `visual_blocks`, `visual_encoder`, `build_model`)

## Context

M3 has 361k parameters and the GPU is mostly idle, so a larger model is affordable. With 63 training
actors, extra capacity may only memorise faces. The question is empirical.

## Decision

- `width` multiplies every size after the encoders: audio Dense, BiLSTM and visual Dense, the fusion
  blocks and the shared trunk. The heads are unchanged. M3 ×2: 967k parameters.
- `visual_blocks: 4` adds a fourth Conv2D block (256 filters) to the deep visual encoder; M3: 723k.
- Both are dev-ladder branches from the step-5 model (0017). They are kept only if they beat it by
  more than the seed sd.

## Consequences

- The FL update size (parameters × 4 bytes) grows with capacity; the federated table reports it.
