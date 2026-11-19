# 0002. Three frozen actor-grouped, sex-stratified splits; seed = split id

- **Status:** Accepted, 18 Nov 2026
- **Where:** `mffer/data/splits.py`, `splits/split_{0,1,2}.json`

## Context

The deployment question is recognising emotion in **unseen people**. If clips of one actor fall in both
train and test, the model can recognise the actor instead of the emotion and the score is inflated.
With 91 actors, one split gives a noisy estimate. Regenerated splits on another machine could silently
differ.

## Decision

Three splits, each 63 / 9 / 19 actors (train / val / test), grouped by actor and stratified by sex. They are
generated once and versioned in `splits/`; `python -m mffer.data.splits` refuses to overwrite them (pre-commit
hooks also leave them untouched). Results are the mean ± sd over the three splits. The weight-initialisation
seed of a run is its split id unless `--seed` is given.

## Consequences

- Every machine and every experiment uses the same actors, so experiments can be compared split by split
  (the federated gap to C2 is paired by split).
- The sd reflects split and seed together. They are not separated (no extra seeds per split).
- Expensive one-split experiments (ladder D1–D4, F3, F4, S1) use split 0 only and are reported as such.
