# 0014. Primary metric: valence macro-F1, then displayed and perceived intensity MAE

- **Status:** Accepted, 19 Nov 2026. Supersedes the selection metric of 0003 and 0004 (val displayed-emotion
  accuracy).
- **Where:** `mffer/metrics.py` (`PRIORITY`, `MONITOR`, `macro_f1`), `mffer/models.py` (`compile_model` logs
  `macro_f1` per classification head), `mffer/train.py` (`DEFAULTS`), `mffer/report.py` and
  `scripts/render_results.py` (column order, gap to C2, by-sex table, curves, `invalid (old monitor)`);
  tested in `tests/test_metrics.py` and `tests/test_pipeline.py`

## Context

The use case is a kiosk that asks whether a customer reacts well or badly to a product. That is a valence
question (positive / neutral / negative), not a six-way emotion question. CREMA-D valence comes from the
displayed emotion, and four of the six emotions map to "negative", so about 68% of clips are negative.
Always answering "negative" (B0) scores 68.3% valence accuracy but only 0.27 macro-F1. The first models
reached 72–73% accuracy, only 4 points above that floor, but 0.50–0.58 macro-F1, about double the floor.
Accuracy hides whether the model finds the rarer positive and neutral customers, which a product test
needs. The intensity heads say how strong the reaction is.

The decision was made on 19 Nov 2026, before C2, the ladder or any federated run had been trained. Only
U1 and C1 had results, so it was not chosen after comparing the main experiments.

## Decision

- **Priority order** (`metrics.PRIORITY`):
  1. displayed-valence macro-F1 (primary);
  2. displayed-intensity MAE (IEO clips only);
  3. perceived-intensity MAE.

  Emotion macro-F1 and accuracy and valence accuracy remain secondary scores.
- **Model selection:** every training mode picks its epoch or round on validation valence macro-F1
  (`metrics.MONITOR = "val_displayed_valence_macro_f1"`), with patience 5 and at most 90 epochs as before.
  The validation value is Keras's `F1Score(average="macro")`, which is tested to equal the sklearn macro-F1
  in `test.json`.
- **Reporting:** tables, the ledger and `results.html` list the metrics in priority order. The federated
  gap to C2 and the by-sex table use valence macro-F1, and the curves figure plots it.
- **Old runs:** the reporter marks a trained run whose `config.yaml` names another monitor as
  `invalid (old monitor)` and never counts it. U1 ×3 and C1 ×3 must be rerun. B0 trains nothing, so it is
  unaffected.

## Consequences

- Selection matches the question the paper asks. Valence is decided by the displayed-emotion label, so
  the emotion heads still drive most of the learning.
- Macro-F1 over 3 classes on 9 validation actors is noisier per epoch than 6-class accuracy, so early
  stopping may fire a little earlier or later. The paper reports which epochs were selected.
- A reviewer may ask whether a separate valence head beats collapsing the emotion head's prediction to
  valence. `predictions.csv` holds both, so this can be computed without new runs.
- Valence here is acted (actors told which emotion to show), not real customer reactions. It stands in for
  them, and the paper states this as a limitation.
