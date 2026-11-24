# 0018. Pool audio over real frames only

- **Status:** Proposed, 24 Nov 2026 (flag `audio_pool: masked`; decided by the dev sweep, 0017)
- **Where:** `mffer/models.py` (`FrameMask`, `MaskedMean`, `audio_encoder`), `tests/test_protocol2.py`

## Context

Clips are zero-padded to 4 s. On average 36% of the 100 MFCC frames are padding (median clip 63 real
frames; p10–p90 49–80). The audio encoder averaged all 100 frames (`GlobalAveragePooling1D`) and its
BatchNorm counted the padding, so a clip's encoding was diluted in proportion to how short it was.

## Decision

`audio_pool: masked`: a frame mask is derived inside the model from the input. The pipeline sets padded
frames to exactly 0 after normalising, and a real frame is never all-zero. The mask is passed to
`BatchNormalization` (masked moments) and to `MaskedMean`, which averages real frames only. The log-mel
encoder (0020) applies the mask, max-pooled to its time resolution, to the time mean. No new model input,
so federated clients and saved models need nothing extra.

## Consequences

- The encoding no longer depends on the amount of padding (tested). Same parameter count.
- A frame zeroed by a time mask (0021) is treated like padding, which is close to SpecAugment's intent.
