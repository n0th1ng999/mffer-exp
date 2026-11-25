# 0007. FL: one local epoch per round, fresh Adam per round, server-side val

- **Status:** Accepted, 19 Nov 2026
- **Where:** `mffer/federated.py` (`client_fit`, `make_strategy`, `RoundLog`)

## Context

The federated experiments must measure the cost of federation, not differences in compute, stopping rule
or model. Flower clients in simulation are stateless, and the kiosk scenario has no shared validation data
on the clients.

## Decision

- FedAvg, weighted by client size, over all clients every round (`fraction_fit = 1`).
- One round = one local epoch on every client, so 90 rounds = 90 centralised epochs of data.
- Each round, the client loads the global weights and recompiles with a **fresh Adam**. No optimiser state
  survives across rounds.
- The server scores the global model on the same val actors as centralised training after every round.
  The same early-stopping rule (0003, 0004) applies, and test is scored once at the val-selected round.
- The global model starts from the same initial weights as C2.
- Clients run one at a time (each gets all CPU cores) in one Flower 1.43 simulation (legacy
  `run_simulation` API, Ray backend).

## Consequences

- Losing Adam's moments between rounds is part of the measured cost of federation. The paper states it.
- Rounds are slightly slower than centralised epochs (about 55 s vs 48 s for M1).
- Server-side val is a simulation convenience. A real deployment would need val data held somewhere.

## Amendment (25 Nov 2026)

"Fresh Adam every round" is now implemented without recompiling: each client's model is compiled once per
worker and `mffer.models.reset_optimizer` sets Adam's step and moments to zero each round (learning rate
kept). Recompiling every round leaked the traced/XLA train functions in the Ray worker (~70 MB per round for
the audio model, ~0.2 GB for M3; V1 F2 split 2 was OOM-killed at round 116); `keras.backend.clear_session`
does not free them. Measured on 15 rounds: worker memory flat at ~1.9 GB (was +62 MB per round), identical
validation metrics; `tests/test_federated.py` checks that reset and recompile give bit-identical weights.
FedProx clients still recompile (their anchor is captured at tracing).
