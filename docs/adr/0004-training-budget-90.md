# 0004. Budget of 90 epochs (or rounds), patience 5, for every configuration

- **Status:** Accepted, 18 Nov 2026 (raised from 30 epochs). Monitored metric superseded by
  [0014](0014-primary-metric-valence-f1.md); budget stands; patience 10 and a 20-epoch minimum since
  [0023](0023-minimum-epochs-before-early-stop.md)
- **Where:** `mffer/train.py` (`DEFAULTS`), applied to federated rounds too

## Context

With a 30-epoch cap, the first runs' best epochs landed at 25–29, so the cap rather than convergence was
ending training. A fair comparison needs the same budget and stopping rule for every configuration.

## Decision

At most 90 epochs, early stopping with patience 5 on val displayed-emotion accuracy, Adam 1e-4 and batch 16,
identical for every configuration. Federated runs get 90 rounds, and one round is one pass over all
training data (see 0007).

## Consequences

- Runs stop on patience, not on the cap (U1 stopped at 30, 43 and 33).
- 30-epoch runs are superseded by the 90-epoch reruns.
- No per-configuration hyperparameter tuning. The paper says so.
