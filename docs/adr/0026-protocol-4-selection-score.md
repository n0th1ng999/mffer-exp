# 0026. Protocol 4: select on the mean of valence and emotion macro-F1 per corpus

- **Status:** Accepted, 25 Nov 2026
- **Where:** `mffer/metrics.py` (`SCORE`, `score`), `mffer/train.py` (`PROTOCOL = 4`, `DEFAULTS["monitor"]`,
  `ValScore`, `val_scorer`), `mffer/federated.py` (`RoundLog`, scorer), `mffer/report.py`
- **Supersedes (selection metric only):** [0014](0014-primary-metric-valence-f1.md); stopping rule from
  [0023](0023-minimum-epochs-before-early-stop.md) unchanged

## Context

V2 pools CREMA-D (7,442 clips) and RAVDESS (1,056). A pooled metric would be decided by CREMA-D. The T1 winner
between A (multi-task CNN) and B (single-task attention model, ADR 0025) must be chosen on what matters for the
kiosk: whether the customer reacts positively (valence) and which emotion, on every population. Decided with
the author on 25 Nov 2026: both F1 scores count, averaged over the corpora.

## Decision

`val_score` = mean over {valence, emotion} of the mean over {CREMA-D, RAVDESS} of the macro-F1 on the
validation actors (displayed labels). It is computed every epoch (centralised, `ValScore` callback) and every
round (federated, server side), logged in epochs.csv with its four parts, and it is the monitor of the
stopping rule (min 20 epochs, patience 10, budget 150) and the rule that picks the T1 winner (mean over
splits, validation only). Test is still scored once, with the best epoch's weights (ADR 0003).

## Consequences

- Protocol-3 and older runs are "invalid (old protocol / old monitor)" in the report; V1 is archived.
- The test report adds the four parts and per-corpus accuracy, so the paper can show both corpora.
- RAVDESS's half rests on 2 validation actors per split (ADR 0024): noisier than CREMA-D's half.
