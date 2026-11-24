# 0019. Weight the valence loss by inverse class frequency, in training only

- **Status:** Proposed, 24 Nov 2026 (flag `balance_valence: true`; decided by the dev sweep, 0017)
- **Where:** `mffer/data/pipeline.py` (`valence_class_weights`, `make_dataset`), `mffer/train.py` and
  `mffer/federated.py` (weights in meta.json), `tests/test_protocol2.py`

## Context

Valence is 68.3% negative, 14.6% neutral and 17.1% positive. The primary metric, which also picks the
epoch (0014), is valence **macro**-F1, which weighs the three classes equally. The loss was plain
cross-entropy, which rewards answering "negative".

## Decision

With `balance_valence: true`, each training clip's valence sample weight is multiplied by
w[c] = n / (3 · n_c), counted on the rows that model trains on. That means the training split
centrally, the client's own rows in FL (like 0008) and the client's rows in L1. The mean weight over
those rows is exactly 1, so the head's loss scale and 0006's coverage compensation are unchanged.
Validation and test datasets keep 0/1 weights, so the monitored val macro-F1 still equals
`metrics.macro_f1`.

## Consequences

- Weights are about 0.49 / 2.28 / 1.95 (negative / neutral / positive) on split 0, recorded in meta.json
  (per client in FL).
- The *training* metrics Keras logs for the valence head are weighted; validation and test are not.
