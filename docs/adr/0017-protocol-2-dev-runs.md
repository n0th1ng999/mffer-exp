# 0017. Protocol 2: versioned runs, and choose model changes on validation-only dev runs

- **Status:** Accepted, 24 Nov 2026
- **Where:** `mffer/train.py` (`PROTOCOL`, `DEFAULTS["protocol"]`, `score_test`, `score_val`, `_seed<k>`
  run names), `mffer/federated.py`, `mffer/report.py` (`invalid (old protocol)`, `dev (val only)`),
  `configs/dev/`, `./run dev`, `scripts/dev_summary.py`

## Context

On the RTX 4080 an M3 epoch takes ~0.8 min (vs ~19 min on the Mac CPU), so model changes can now be
tested before the paper runs. Those were not yet run under the current monitor (0014). Two bugs found on
24 Nov also change every run: the shuffle seed (0016, bug-fix note) and resume after a crash between an
epoch's row and its checkpoint (resume points `last-<n>.keras` / `last-<n>.weights.h5`,
`train.load_history`). Choosing model changes must not touch the test actors (0003).

## Decision

- Every run records `protocol` in its config.yaml; this repository is at protocol **2**. The reporter
  marks finished trained runs of an older protocol (no field = 1) "invalid (old protocol)". B0 trains
  nothing and is exempt.
- Model changes (0018–0022) are flags of `ModelSpec`, off by default. They are chosen on **dev runs**:
  `configs/dev/dev_<model>_<step>_<change>.yaml`, split 0, seeds 0 1 2, `score_test: false`. A dev run
  stops after validation, writes `val.json` instead of `test.json`, and never counts.
- The dev ladder adds one change per step (U1 and M3; steps 6–7 branch from step 5).
  `scripts/dev_summary.py` gives mean ± sd over seeds and the gain per step. A change is kept when its
  gain in val valence macro-F1 clearly exceeds the seed sd. Kept flags become the `ModelSpec` defaults
  (so every preset inherits them), recorded by setting each ADR's status.
- A run with a non-default `--seed` is named `<config>_seed<k>` so seed runs never share a directory.

## Consequences

- All protocol-1 runs (U1 and C1 of 18 Nov, the 20 Nov lab runs) are invalid for the paper; B0 stays.
- Split 0's validation actors are used for model selection; the paper says so. Test stays untouched until
  the paper runs.
- The paper reports the final flags as the protocol, and the dev ladder as the justification (supplement).
