# 0006. Missing labels masked per head; loss weight divided by label coverage

- **Status:** Accepted, 18 Nov 2026
- **Where:** `mffer/data/pipeline.py` (sample weights, `label_coverage`), `mffer/models.py` (`compile_model`)

## Context

Not every clip has every label. Displayed intensity exists only for sentence IEO (1,365 clips). Perceived
emotion is undefined for 420 tied clips. Dropping such clips would discard the other four labels. Masking
alone would shrink a head's effective weight in proportion to its coverage.

## Decision

A missing label gets sample weight 0 for that head only. Each head's configured loss weight is divided by
its label coverage on the training rows (each federated client uses its own coverage), so the configured
weights (1.0 per classification head, 0.2 per intensity head) hold in expectation.

## Consequences

- Every clip contributes to every head it has a label for.
- Metrics for a masked head are computed only on clips that have that label.
