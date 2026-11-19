# 0010. L1 trains on exactly F2's client rows; reported as the mean of clients

- **Status:** Accepted, 19 Nov 2026
- **Where:** `mffer/data/splits.py` (`client_rows`), `mffer/train.py` (`--client`), `mffer/report.py` (L1 aggregation); tested in `tests/test_pipeline.py`

## Context

L1 asks whether federating beats not collaborating. The comparison is only fair if each local-only model
sees exactly the data its federated counterpart holds.

## Decision

`splits.client_rows(clips, split, n_clients, mode)` is the single definition of a client's data
(partition seed = split id), used by both the federated runner and `mffer.train --client k`. L1 trains each
client with the centralised trainer restricted to its rows, scored on the same val and test actors. A split
counts only when both clients finished, and its score is the mean of the two clients.

## Consequences

- F2 vs L1 isolates the effect of aggregation.
- An L1 split needs two runs, and the reporter reports nothing for a split with one client missing.
