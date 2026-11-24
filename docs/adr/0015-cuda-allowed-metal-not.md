# 0015. Train on CUDA where available; Metal stays hidden

- **Status:** Accepted, 19 Nov 2026. Supersedes the "CPU only everywhere" part of 0001; that the Metal GPU
  is invalid stands. CUDA checked against the CPU on 20 Nov 2026: agrees (see Outcome). The "not
  deterministic" consequence is superseded by [0016](0016-deterministic-ops.md)
- **Where:** `mffer/device.py` (`configure`, `current`, `names`, `check_resume`), `mffer/train.py`,
  `mffer/federated.py` (client device report, `aggregate_fit` check, Ray `num_gpus`), `mffer/report.py`
  (status), `scripts/smoke_test.py`; tested in `tests/test_report.py`

## Context

0001 hid every GPU because the Apple Metal backend diverged. The cause was located in tensorflow-metal, not
in the model, so it gives no reason to distrust CUDA. A second machine with an RTX 4080 is available, and
M3 on the CPU takes up to about 9.5 h per split, which caps how many runs fit before 6 Dec.

## Decision

- One policy in `mffer.device`, applied by every entry point before TensorFlow uses a device:
  - **macOS:** the GPU is hidden and training runs on the CPU.
  - **Linux / WSL2 with an NVIDIA GPU:** CUDA, with memory growth on, so the Flower server and client
    processes can share the card.
  - **`MFFER_CPU=1`:** forces the CPU anywhere.
- Every run records `device` (`cpu` / `cuda` / `metal`) and the GPU model in `meta.json`.
- A run must be resumed on the device it started on (`check_resume`).
- Federated clients report their device every round, and the round fails unless all of them used the run's
  device (this extends 0009). On CUDA, each client asks Ray for the one GPU, so clients still run one at a time.
- The reporter counts CPU and CUDA runs. Metal runs are `invalid (GPU)`. Runs from before this record have
  no `device` field, so any GPU they list was Metal.

## Consequences

- **Before any paper run on CUDA,** run the check in the README ("First job on a CUDA machine"): U1, split
  0, 12 epochs, same seed, CPU vs CUDA, with `scripts/lab_stability.py`. If the curves agree within about a
  point and the loss keeps falling, add the result to this record. If CUDA diverges, train there with
  `MFFER_CPU=1` and supersede this record.
- CUDA training is not bit-for-bit deterministic (`enable_op_determinism` is off, to keep its speed and
  avoid ops without a deterministic kernel), so reruns differ by run-to-run noise. Results are already the
  mean ± sd over three splits.
- Experiments may come from different machines and devices. The ledger has a Device column, and the paper
  states which hardware trained which experiment. Per-epoch times are only comparable within one device.
- The CUDA path (memory growth, Ray GPU assignment under WSL2) has not been run yet. The smoke test on the
  new machine is its first test.

## Outcome of the CUDA check (20 Nov 2026)

Machine: Windows 11, WSL2 (Ubuntu), NVIDIA GeForce RTX 4080 16 GB (driver 560.94), AMD Ryzen 9 7950X;
TensorFlow 2.19.1, Keras 3.15.1. `scripts/lab_stability.py --preset audio_only --split 0 --epochs 12`,
`--cpu` vs CUDA, same seed, before op determinism (0016) was switched on:

| Epoch | CPU val displayed acc | CUDA val displayed acc | Train loss (both) |
|---|---|---|---|
| 1 | 0.369 | 0.367 | 4.30 |
| 4 | 0.438 | 0.436 | 3.61 |
| 8 | 0.474 | 0.481 | 3.44 |
| 12 | 0.501 | 0.499 | 3.31 |

Largest gap 0.7 points; the loss falls every epoch on both and matches to two decimals throughout, unlike
Metal, which diverged after 3–4 epochs. CUDA runs count. Smoke test on the same machine: M3 0.157 s/step on
CUDA vs 1.46 s/step on the CPU (about 9×). The federated CUDA path (Ray `num_gpus`) was then exercised by the
end-to-end check in 0016.
