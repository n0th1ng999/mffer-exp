# 0001. Train on the CPU only; GPU runs are invalid

- **Status:** Accepted, 18 Nov 2026. CPU-only-everywhere superseded by [0015](0015-cuda-allowed-metal-not.md)
  (CUDA allowed); Metal stays invalid
- **Where:** `mffer/train.py` (`main` hides the GPU), `mffer/federated.py` (`hide_gpu`, per-round `n_gpus` check), `mffer/report.py` (status `invalid (GPU)`)

## Context

On the Apple M4 Pro GPU (tensorflow-metal 1.2.0, TF 2.18 and 2.19) the same code, data and seed diverged
after 3–4 epochs, while the CPU trained cleanly. `scripts/lab_stability.py` reproduces it: it trains
one configuration under variants (CPU/GPU, learning rate, clipnorm, no BatchNorm) and prints the curves.
The divergence is in the Metal backend, not in the model.

## Decision

Every training (centralised, local-only, federated clients) runs on the CPU with the GPU hidden. Each run
records its visible GPUs in `meta.json` (`"gpu": []` = CPU). Federated clients report `n_gpus` every round,
and a round fails if any client saw one. The reporter never counts a run with a non-empty `gpu` list.

## Consequences

- Runs that used the GPU before 18 Nov are invalid and excluded automatically.
- Training is slower (M3: up to about 9.5 h per split), which caps how many configurations and seeds fit
  before the deadline.
- The paper reports CPU training. Reproducibility across machines is to within multithreaded CPU
  summation order, not bit for bit.
