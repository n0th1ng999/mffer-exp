"""Which device trains, decided in one place (ADR 0015).

- macOS: CPU only. The Metal GPU (tensorflow-metal 1.2.0, TF 2.18 and 2.19) diverges after 3-4 epochs where
  the CPU trains cleanly with the same code, data and seed (scripts/lab_stability.py, ADR 0001).
- Linux / WSL2 with an NVIDIA GPU: CUDA, with memory growth so the Flower server and client processes can
  share the card. `MFFER_CPU=1` forces the CPU (e.g. to compare a CUDA run with a CPU run).

Every run records its device in meta.json (`device`: cpu | cuda | metal); the reporter never counts metal,
and a run must be resumed on the device it started on.

Op determinism is always on (ADR 0016): the same code, data, seed, device and software give bit-identical
weights, on the CPU and on CUDA. It does not make CPU and CUDA (or two different CPUs) agree bit for bit.
"""
import os
import sys

import tensorflow as tf
from tensorflow.python.framework import config as tf_config  # TF 2.19 has only the setter in tf.config


def configure() -> str:
    """Apply the policy. Call before TensorFlow uses a device in this process; returns `current()`."""
    # Deterministic kernels (cuDNN, reductions, tf.data); an op without one raises instead of running.
    tf.config.experimental.enable_op_determinism()
    gpus = tf.config.list_physical_devices("GPU")
    try:
        if sys.platform == "darwin" or os.environ.get("MFFER_CPU") == "1":
            tf.config.set_visible_devices([], "GPU")
        else:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:   # devices already initialised in this process: current() reports what is in use
        pass
    return current()


def deterministic() -> bool:
    return tf_config.is_op_determinism_enabled()


def current() -> str:
    if not tf.config.get_visible_devices("GPU"):
        return "cpu"
    return "metal" if sys.platform == "darwin" else "cuda" if tf.test.is_built_with_cuda() else "gpu"


def names() -> list[str]:
    """Visible GPUs by model name where TensorFlow knows it (e.g. "NVIDIA GeForce RTX 4080")."""
    return [tf.config.experimental.get_device_details(d).get("device_name", d.name)
            for d in tf.config.get_visible_devices("GPU")]


def check_resume(meta: dict) -> None:
    """A run continues on the device it started on (runs from before 19 Nov have no `device`: CPU)."""
    started = meta.setdefault("device", "metal" if meta.get("gpu") else "cpu")
    if started != current():
        raise SystemExit(f"run started on {started}, this process uses {current()}: resume it on the same "
                         f"device (MFFER_CPU=1 forces the CPU)")
