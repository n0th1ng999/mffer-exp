# 0008. FL: audio normalisation per client, pooled exactly on the server

- **Status:** Accepted, 19 Nov 2026
- **Where:** `mffer/data/pipeline.py` (`fit_stats`, `AudioStats.combine`), `mffer/federated.py` (`run`)

## Context

Centralised training normalises MFCCs with train-split statistics. Under federation no party sees all
training audio, but the server must still normalise val and test inputs to score the global model.

## Decision

Each client computes its own MFCC mean and std and normalises its own data with them. The clients'
summary statistics (13 means, 13 stds and its frame count) are combined exactly into pooled statistics,
which the server uses for val and test. Both levels are saved: `audio_stats_c<k>.json` and `audio_stats.json`.

## Consequences

- Raw audio never leaves a client; only those 27 summary numbers per client do.
- Clients train on slightly different input scalings, part of the realistic non-IID cost.
