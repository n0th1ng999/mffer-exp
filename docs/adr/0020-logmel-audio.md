# 0020. Log-mel audio features with a 2-D convolutional encoder

- **Status:** Proposed, 24 Nov 2026 (flag `audio_features: logmel`; decided by the dev sweep, 0017)
- **Where:** `mffer/data/features.py` (`--audio logmel` → `../data/processed/logmel.h5`),
  `mffer/data/pipeline.py` (`Cremad.audio`), `mffer/models.py` (`AUDIO_SHAPES`, `logmel_encoder`)

## Context

The audio input was 13 MFCCs at a 40 ms hop: little spectral detail and coarse timing for prosody, which
carries much of the emotion in CREMA-D's scripted sentences. Pretrained audio models are out of scope;
the goal is the best model learned from CREMA-D alone.

## Decision

- Features: 16 kHz, 64 mel bands, n_fft 400 (25 ms), hop 160 (10 ms), ln(power + 1e-6), 400 frames = 4 s,
  zero-padded; stored in a separate `logmel.h5` (0.76 GB, ~1 min with 24 workers), rows checked against
  clips.csv like `cremad.h5`. Built from the pinned CREMA-D commit (only `AudioWAV` is needed).
- Normalisation as before: per band, training rows only (`fit_stats(..., "logmel")`, per client in FL).
- Encoder: 3 × [Conv2D(32/64/128, 3×3) + BN + MaxPool 2] over (time, mel) → mean over mel → mean over
  time (masked with 0018) → Dense 64·width.

## Consequences

- U1: 60k → 154k parameters; M3: 361k → 455k. `logmel.h5` must exist on any machine that trains with it
  (README: Reproduce).
