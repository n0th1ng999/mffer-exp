"""Environment check on random tensors (no dataset needed).

1. Reports the training device (mffer.device: CPU on macOS, CUDA on Linux with an NVIDIA GPU).
2. M1 and M3 train for a few steps without NaNs; seconds per step give a first timing estimate.
3. A Flower simulation runs 2 FedAvg rounds with 2 clients.
"""
import argparse
import math
import time

import numpy as np
import tensorflow as tf

from mffer import device
from mffer.models import AUDIO_SHAPE, DEMO_DIM, HEADS, PRESETS, TEXT_DIM, VIDEO_SHAPE, build_model

BATCH = 16


def random_batch(spec, n=BATCH):
    rng = np.random.default_rng(0)
    x = {"audio": rng.standard_normal((n, *AUDIO_SHAPE), dtype=np.float32),
         "video": rng.random((n, *VIDEO_SHAPE), dtype=np.float32)}
    if spec.text:
        x["text"] = rng.standard_normal((n, TEXT_DIM), dtype=np.float32)
    if spec.demographics:
        x["demographics"] = rng.random((n, DEMO_DIM), dtype=np.float32)
    y = {}
    for name in spec.heads:
        units = HEADS[name][0]
        y[name] = (np.eye(units, dtype=np.float32)[rng.integers(0, units, n)] if units > 1
                   else rng.random((n, 1), dtype=np.float32))
    return x, y


def check_training(preset: str, steps: int) -> None:
    spec = PRESETS[preset]
    model = build_model(spec)
    x, y = random_batch(spec)
    model.train_on_batch(x, y)                       # first step compiles the graph
    start = time.perf_counter()
    for _ in range(steps):
        loss = model.train_on_batch(x, y, return_dict=True)["loss"]
    per_step = (time.perf_counter() - start) / steps
    assert math.isfinite(loss), f"{preset}: loss is {loss}"
    # ~70% of 7,442 clips are training clips under the actor split.
    epoch_min = per_step * (0.7 * 7442 / BATCH) / 60
    print(f"{preset}: {model.count_params():,} params, loss {loss:.3f}, {per_step:.3f} s/step "
          f"-> about {epoch_min:.0f} min/epoch of pure compute (data loading excluded)")


def check_flower() -> None:
    from flwr.client import ClientApp, NumPyClient
    from flwr.common import Context
    from flwr.server import ServerApp, ServerAppComponents, ServerConfig
    from flwr.server.strategy import FedAvg
    from flwr.simulation import run_simulation

    def tiny_model():
        model = tf.keras.Sequential([tf.keras.Input((4,)), tf.keras.layers.Dense(2, activation="softmax")])
        model.compile("adam", "categorical_crossentropy")
        return model

    class Client(NumPyClient):
        def __init__(self):
            self.model = tiny_model()
            rng = np.random.default_rng()
            self.x = rng.random((32, 4), dtype=np.float32)
            self.y = np.eye(2, dtype=np.float32)[rng.integers(0, 2, 32)]

        def fit(self, parameters, config):
            self.model.set_weights(parameters)
            self.model.fit(self.x, self.y, epochs=1, verbose=0)
            return self.model.get_weights(), len(self.x), {}

    def server_fn(context: Context):
        initial = tiny_model().get_weights()
        from flwr.common import ndarrays_to_parameters
        strategy = FedAvg(fraction_evaluate=0.0, min_fit_clients=2, min_available_clients=2,
                          initial_parameters=ndarrays_to_parameters(initial))
        return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=2))

    run_simulation(server_app=ServerApp(server_fn=server_fn),
                   client_app=ClientApp(client_fn=lambda context: Client().to_client()),
                   num_supernodes=2, verbose_logging=False)
    print("flower: 2 rounds with 2 clients completed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--skip-flower", action="store_true")
    args = parser.parse_args()

    print(f"GPUs found: {tf.config.list_physical_devices('GPU')}")
    print(f"training device: {device.configure()} {device.names()}")   # cuda expected on an NVIDIA machine
    for preset in ("m1", "m3"):
        check_training(preset, args.steps)
    if not args.skip_flower:
        check_flower()
    print("smoke test passed")


if __name__ == "__main__":
    main()
