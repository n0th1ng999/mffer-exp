# 0009. FL: a client failure or a visible GPU fails the run

- **Status:** Accepted, 19 Nov 2026
- **Where:** `mffer/federated.py` (`aggregate_fit`, `accept_failures=False`, post-simulation checks)

## Context

Flower can silently average the surviving clients when one fails. In a 2-client run that turns the round
into single-client training without any trace in the scores. A client that quietly used the GPU would
break 0001.

## Decision

`aggregate_fit` raises if any client failed, if fewer than all clients returned, or if any client reports
a visible GPU. After the simulation, the run must be marked finished and `best.weights.h5` must exist
before test is scored. A failed run exits non-zero and can be resumed from its last completed round.

## Consequences

- Every counted federated run averaged all of its clients in every round.
- One flaky client costs a resume, not a silently wrong result.
