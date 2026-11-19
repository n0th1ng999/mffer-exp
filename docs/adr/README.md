# Architecture Decision Records

One file per decision that shapes the experiments or the code, in Michael Nygard's format: **Context**
(the forces at play), **Decision**, **Consequences** (what becomes easier, harder, or must be reported).
They answer "why is it done this way?" for reviewers, co-authors and the paper's Methods section.

Rules:
- A record is never edited to change its decision. A new record supersedes it, and the old one's status
  becomes `Superseded by NNNN`.
- Any change to the architecture or the experimental protocol adds or supersedes a record in the same commit.
- Copy `template.md` for a new record; number sequentially.

| # | Decision | Status |
|---|---|---|
| [0001](0001-cpu-only-training.md) | Train on the CPU only; GPU runs are invalid | Accepted, 18 Nov 2026 |
| [0002](0002-frozen-actor-splits.md) | Three frozen actor-grouped, sex-stratified splits; seed = split id | Accepted, 18 Nov 2026 |
| [0003](0003-test-once-val-selected.md) | Early stopping on val; test scored once with the best weights | Accepted, 18 Nov 2026 |
| [0004](0004-training-budget-90.md) | Budget of 90 epochs (or rounds), patience 5, for every configuration | Accepted, 18 Nov 2026 |
| [0005](0005-one-model-builder.md) | One model builder; every configuration is a preset of flags | Accepted, 18 Nov 2026 |
| [0006](0006-masked-heads-coverage.md) | Missing labels masked per head; loss weight divided by label coverage | Accepted, 18 Nov 2026 |
| [0007](0007-fl-round-protocol.md) | FL: one local epoch per round, fresh Adam per round, server-side val | Accepted, 19 Nov 2026 |
| [0008](0008-per-client-normalisation.md) | FL: audio normalisation per client, pooled exactly on the server | Accepted, 19 Nov 2026 |
| [0009](0009-fail-whole-round.md) | FL: a client failure or a visible GPU fails the run | Accepted, 19 Nov 2026 |
| [0010](0010-local-only-same-rows.md) | L1 trains on exactly F2's client rows; reported as the mean of clients | Accepted, 19 Nov 2026 |
| [0011](0011-generated-paper-numbers.md) | Paper numbers come only from counted run directories | Accepted, 19 Nov 2026 |
| [0012](0012-fedprox-untuned-mu.md) | FedProx with μ = 0.01, not tuned | Accepted, 19 Nov 2026 |
| [0013](0013-lint-no-formatter.md) | Lint with ruff, no auto-formatter; hooks outside the experiment env | Accepted, 19 Nov 2026 |
