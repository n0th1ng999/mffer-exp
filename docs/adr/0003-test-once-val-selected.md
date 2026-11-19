# 0003. Early stopping on val; test scored once with the best weights

- **Status:** Accepted, 18 Nov 2026. Selection metric superseded by [0014](0014-primary-metric-valence-f1.md)
  (now val valence macro-F1); the test-once rule stands
- **Where:** `mffer/train.py` (`EpochCheckpoint`, `score_test`), `mffer/federated.py` (`RoundLog`)

## Context

Choosing the epoch, or anything else, by looking at test scores leaks the test set into model selection,
and the reported score becomes optimistic.

## Decision

The monitored quantity is validation displayed-emotion accuracy. The best epoch's (or round's) weights are
saved as `best.weights.h5`. Test predictions are computed exactly once, at the end of the run, with those
weights, and written to `predictions.csv` and `test.json`. Resuming a finished run only re-scores the same
weights.

## Consequences

- Test numbers are untouched by selection, so they can be compared across experiments.
- Only the displayed-emotion head chooses the epoch, so the other heads may be scored at an epoch that is
  not their best. The paper states this.
