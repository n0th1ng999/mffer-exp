# 0012. FedProx with μ = 0.01, not tuned

- **Status:** Accepted, 19 Nov 2026
- **Where:** `mffer/federated.py` (`ProxModel`), `configs/runs/f4_fedprox.yaml`; tested in `tests/test_federated.py`

## Context

F4 tests whether a method designed for heterogeneous clients helps the kiosk case. Tuning μ on val would
cost several M3 federated runs (hours each) before the deadline.

## Decision

Clients add (μ/2)·‖w − w_global‖² to their training loss (not to the evaluation loss), with μ = 0.01, a
common default. Aggregation stays FedAvg. With μ = 0 the client trains bit-identically to plain FedAvg
(tested).

## Consequences

- A null result (F4 ≈ F2) is expected with 2 clients and 1 local epoch, and is reported as such.
- The paper states that μ was not tuned.
