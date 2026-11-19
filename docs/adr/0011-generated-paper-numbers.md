# 0011. Paper numbers come only from counted run directories

- **Status:** Accepted, 19 Nov 2026
- **Where:** `mffer/report.py`, `runs/<config>-s<split>-<stamp>/`, `../mffer-worldcist27/generated/`

## Context

Hand-copied numbers drift from the runs, and with reruns, superseded and invalid runs it becomes unclear
which run a number came from.

## Decision

Every run writes a self-describing directory: config, split, seed, git version (`git describe --dirty`),
hardware, `epochs.csv`, test predictions and metrics. The reporter counts, per (configuration, split), the
latest finished CPU run. Other runs are `superseded`, `invalid (GPU)` or `incomplete`. It writes the
results ledger and the LaTeX tables and figure, and each table lists its source run IDs in comments.

## Consequences

- Rerunning `python -m mffer.report` regenerates every number in the paper.
- Code must be committed before launching a run, or its recorded version ends in `-dirty` and cannot be
  rebuilt exactly.
