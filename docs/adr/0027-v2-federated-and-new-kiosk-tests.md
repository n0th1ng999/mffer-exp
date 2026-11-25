# 0027. V2 tests after T1: FedAvg per corpus (T3a), random actor clients (T3b), new kiosk (T5)

- **Status:** Accepted, 25 Nov 2026
- **Where:** `configs/v2/t3a_*.yaml`, `t3b_k{2,5,10}_*.yaml`, `t5_*_to_*_*.yaml`; `mffer/train.py` (`corpus_rows`,
  config key `corpus`); `mffer/data/splits.py` (partitions `corpus`, `actor`)

## Context

The V2 goal (author, 25 Nov 2026): show the models on the kiosk use case with federated learning (GDPR: raw
faces and voices stay on the kiosk), under non-IID data. Decided with the author on 25 Nov: T2 (local only),
T4 (FedProx, FedBN, per-client normalisation) are cut; only the T1 winner goes on; T6 costs are reported for
every run, not as a separate test; T3b runs last.

## Decision

All runs use the V2 data (ADR 0024), protocol 4 (ADR 0026), splits 0-2, and the T1 winner's preset. Configs
exist for both models (`_a`, `_b`); the loser's are deleted once T1 is decided.

- **T3a** FedAvg, `partition: corpus`: client 1 = CREMA-D's training actors (63, 5,152 clips), client 2 =
  RAVDESS's (18, 792): quantity, label, feature and concept skew as measured in the EDA. Same val and test
  actors as T1: T3a - T1 is the cost of keeping each kiosk's data on the kiosk.
- **T3b** FedAvg, `partition: actor`, k = 2, 5, 10: whole actors of both corpora dealt at random within sex
  (seed = split id); near-IID control for T3a (T3b k=2 - T3a isolates non-IID) and the cost of more, smaller
  kiosks (k).
- **T5** new kiosk, centralised: `corpus: {train: X, test: Y}` trains and selects on corpus X's actors of the
  split only and tests on corpus Y's test actors (the same actors as T1's Y test). Both directions. T1's Y score
  minus T5's is the price of having no data at all from that kiosk.

## Consequences

- T5 RAVDESS -> CREMA-D selects on 2 validation actors (88 clips) and trains on 792 clips: expected weak and
  noisy; it is the "tiny kiosk" end of the scale.
- In T3b, k = 10 gives clients of 7-9 actors (~500-700 clips); one client of split 0 has no RAVDESS actor.
- Each federated run records parameters, bytes per update and rounds, so T6 reports MB until the best round.
