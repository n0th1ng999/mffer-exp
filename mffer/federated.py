"""Federated training of one configuration on one split (Flower simulation, FedAvg).

    python -m mffer.federated configs/runs/f2_actor.yaml --split 0
    python -m mffer.federated --resume runs/f2_actor-s0-20261124-2310      # continue after a crash

A config adds `federated: {clients: N, partition: iid | actor}` to the usual run config. The protocol matches
`mffer.train`: same model, Adam 1e-4, batch 16, at most `max_epochs` rounds, early stopping with `patience`
on `monitor` (never before `min_epochs`, ADR 0023), test scored once with the best round's weights. One round = one local epoch on every client,
i.e. one pass over all training rows, so rounds and centralised epochs cost the same.

- Clients: `splits.client_rows` (seed = split id), so client k of F2 holds the rows local-only client k
  (L1, `mffer.train --client k`) trains on. Each client normalises audio with statistics of its own rows
  and compensates loss weights with its own label coverage. Clients keep no state between rounds: every
  round starts from the global weights with a fresh Adam (standard FedAvg; makes resume exact).
- Server: FedAvg weighted by client size. After every round it scores the global model on the val actors,
  with the clients' audio statistics pooled exactly (`AudioStats.combine`: 27 numbers per client), and
  saves the global weights (resume point) and the best round's weights.
- FedProx (F4): `federated: {..., mu: 0.01}` adds (mu/2)·||w - w_global||² to every client's training
  loss, which keeps a client from drifting far from the global model in its local epoch. The server side
  is plain FedAvg (FedProx changes only the client). Without `mu`, clients run plain `model.fit`.
- One Flower simulation process; clients run one at a time, on the run's device (mffer.device: CPU on
  macOS, CUDA on Linux with an NVIDIA GPU). Every round checks that all clients reported and that every
  client trained on that same device.

Run directory: the files of `mffer.train` (epochs.csv has one row per round, `epoch` = round - 1), plus
`audio_stats_c<k>.json` per client.
"""
import argparse
import json
import os
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from mffer import device
from mffer.config import load_run_config
from mffer.data.pipeline import AudioStats, Cremad, fit_stats, label_coverage, make_dataset, valence_class_weights
from mffer.data.splits import client_rows, rows
from mffer.models import build_model, compile_model, derive_seed, reseed, spec_from_config
from mffer.train import (
    DEFAULTS,
    load_history,
    new_run,
    print_result,
    save_atomic,
    save_checkpoint,
    score_test,
    score_val,
    should_stop,
)

# ---------------------------------------------------------------- client (runs in a Flower/Ray worker)

_cache: dict = {}   # per worker process: the dataset and one model per run directory


class ProxModel(keras.Model):
    """The same functional model (shared layers and weights) with the FedProx proximal term added to
    the training loss. `compute_loss` is Keras's hook for this; evaluation calls it with training=False."""

    def set_anchor(self, mu: float, weights: list[np.ndarray]) -> None:
        self.prox = (mu, [tf.constant(w) for w in weights])   # constants, so not counted as model weights

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, training=True):
        loss = super().compute_loss(x, y, y_pred, sample_weight, training=training)
        mu, anchor = self.prox
        if training and mu:
            loss += 0.5 * mu * tf.add_n([tf.reduce_sum(tf.square(v - a))
                                         for v, a in zip(self.trainable_weights, anchor, strict=True)])
        return loss


def _client_state(run_dir: Path) -> dict:
    if "data" not in _cache:
        _cache["data"] = Cremad()
    if _cache.get("run_dir") != run_dir:
        config = yaml.safe_load((run_dir / "config.yaml").read_text())
        spec = spec_from_config(config["model"])
        model = build_model(spec)
        if config["federated"].get("mu"):
            model = ProxModel(model.input, model.output)
        _cache.update(run_dir=run_dir, config=config, spec=spec, model=model,
                      meta=json.loads((run_dir / "meta.json").read_text()))
    return _cache


def client_fit(run_dir: Path, client: int, parameters: list[np.ndarray], round_: int) -> tuple:
    state = _client_state(run_dir)
    data, config, spec, model, meta = (state[k] for k in ("data", "config", "spec", "model", "meta"))
    fed = config["federated"]
    own = client_rows(data.clips, meta["split"], fed["clients"], fed["partition"])[client]
    stats = AudioStats.load(run_dir / f"audio_stats_c{client}.json")

    start = time.perf_counter()
    model.set_weights(parameters)
    compile_model(model, spec, label_coverage(data, own, spec))   # fresh Adam every round
    if isinstance(model, ProxModel):   # anchor = this round's global weights, trainable ones only
        model.set_anchor(fed["mu"], [w.numpy() for w in model.trainable_weights])
    seed = derive_seed(meta["seed"], client, round_)
    keras.utils.set_random_seed(seed)
    reseed(model, seed)   # dropout from (seed, client, round) only, whichever worker process runs it (ADR 0016)
    history = model.fit(make_dataset(data, own, spec, stats, training=True, seed=seed,
                                     batch_size=config["batch_size"]),
                        epochs=1, verbose=0, shuffle=False)
    seconds = time.perf_counter() - start
    metrics = {k: float(v[-1]) for k, v in history.history.items()}
    print(f"round {round_} client {client}: {len(own)} clips, loss {metrics['loss']:.3f}, {seconds:.0f} s",
          flush=True)
    return model.get_weights(), len(own), {**metrics, "seconds": seconds, "client": client,
                                            "device": device.current()}


def make_client_app(run_dir: Path):
    from flwr.client import ClientApp, NumPyClient

    class Client(NumPyClient):
        def __init__(self, client: int):
            self.client = client

        def fit(self, parameters, config):
            return client_fit(run_dir, self.client, parameters, int(config["round"]))

    def client_fn(context):
        device.configure()
        return Client(int(context.node_config["partition-id"])).to_client()

    return ClientApp(client_fn=client_fn)


# ---------------------------------------------------------------- server (main process)

class RoundLog:
    """Server-side bookkeeping, the federated counterpart of train.EpochCheckpoint: after each round append
    to epochs.csv, keep the best round's weights, then save the global weights as the resume point
    (`last-<round>.weights.h5`; see train.load_history), stop by train.should_stop.
    State is rebuilt from epochs.csv on resume."""

    def __init__(self, run_dir: Path, model: keras.Model, val: tf.data.Dataset, config: dict):
        self.run_dir, self.model, self.val = run_dir, model, val
        self.config, self.monitor = config, config["monitor"]
        self.file = run_dir / "epochs.csv"
        history, self.last = load_history(run_dir, ".weights.h5")
        self.done = len(history)
        self.best = history[self.monitor].max() if self.done else -np.inf
        self.best_round = int(history[self.monitor].idxmax()) if self.done else -1   # 0-based, as epochs.csv
        self.since_best = self.done - 1 - self.best_round if self.done else 0
        self.train_metrics: dict = {}
        self.started = time.perf_counter()

    @property
    def finished(self) -> bool:
        return should_stop(self.done, self.since_best, self.config)

    def end_round(self, weights: list[np.ndarray]) -> None:
        self.model.set_weights(weights)
        val = self.model.evaluate(self.val, verbose=0, return_dict=True)
        row = {"epoch": self.done, **self.train_metrics, **{f"val_{k}": float(v) for k, v in val.items()},
               "round_seconds": time.perf_counter() - self.started}
        pd.DataFrame([row]).to_csv(self.file, mode="a", header=not self.file.exists(), index=False)
        if row[self.monitor] > self.best:
            self.best, self.best_round, self.since_best = row[self.monitor], self.done, 0
            save_atomic(self.run_dir / "best.weights.h5", self.model.save_weights)
        else:
            self.since_best += 1
        self.done += 1
        save_checkpoint(self.run_dir, self.done, ".weights.h5", self.model.save_weights)   # after the row
        print(f"round {self.done}: {self.monitor} {row[self.monitor]:.4f} (best {self.best:.4f}, round "
              f"{self.best_round + 1}), {row['round_seconds']:.0f} s", flush=True)
        self.started = time.perf_counter()


def make_strategy(log: RoundLog, n_clients: int, initial: list[np.ndarray], expected_device: str):
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server.strategy import FedAvg

    class Strategy(FedAvg):
        """FedAvg that (1) fails the run if any client fails, instead of averaging the survivors,
        (2) asserts every client trained on the run's device, (3) evaluates on val and stops early via `log`."""

        def configure_fit(self, server_round, parameters, client_manager):
            return [] if log.finished else super().configure_fit(server_round, parameters, client_manager)

        def aggregate_fit(self, server_round, results, failures):
            if failures or len(results) != n_clients:
                first = str(failures[0])[-300:] if failures else ""   # Ray's full trace is printed above
                raise RuntimeError(f"round {server_round}: {len(results)} of {n_clients} clients reported; "
                                   f"{len(failures)} failed: ...{first}")
            # Client order, not arrival order: FedAvg's float sums then add up in the same order every run.
            results = sorted(results, key=lambda result: result[1].metrics["client"])
            metrics = [(r.num_examples, r.metrics) for _, r in results]
            devices = [m["device"] for _, m in metrics]
            if set(devices) != {expected_device}:
                raise RuntimeError(f"clients trained on {devices}, the run uses {expected_device}")
            n = sum(k for k, _ in metrics)
            log.train_metrics = {key: sum(k * m[key] for k, m in metrics) / n
                                 for key in metrics[0][1] if key not in ("device", "seconds", "client")}
            log.train_metrics["client_seconds"] = sum(m["seconds"] for _, m in metrics)
            return super().aggregate_fit(server_round, results, failures)

        def evaluate(self, server_round, parameters):
            if server_round == 0 or log.finished:   # no round-0 row: epochs.csv rows match centralised epochs
                return None
            log.end_round(parameters_to_ndarrays(parameters))
            return 0.0, {}

    return Strategy(fraction_fit=1.0, fraction_evaluate=0.0, min_fit_clients=n_clients,
                    min_available_clients=n_clients, accept_failures=False,
                    on_fit_config_fn=lambda server_round: {"round": log.done + 1},
                    initial_parameters=ndarrays_to_parameters(initial))


def run(run_dir: Path) -> dict:
    from flwr.server import ServerApp, ServerAppComponents, ServerConfig
    from flwr.simulation import run_simulation

    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    meta = json.loads((run_dir / "meta.json").read_text())
    device.check_resume(meta)
    fed = config["federated"]
    keras.utils.set_random_seed(meta["seed"])
    spec = spec_from_config(config["model"])
    data = Cremad()
    train, val, test = (rows(data.clips, meta["split"], p) for p in ("train", "val", "test"))
    clients = client_rows(data.clips, meta["split"], fed["clients"], fed["partition"])

    # Each client's statistics come from its own rows; the server only pools the summaries.
    for k, own in enumerate(clients):
        if not (run_dir / f"audio_stats_c{k}.json").exists():
            fit_stats(data, own, spec.audio_features).save(run_dir / f"audio_stats_c{k}.json")
    stats = AudioStats.combine([AudioStats.load(run_dir / f"audio_stats_c{k}.json") for k in range(len(clients))])
    stats.save(run_dir / "audio_stats.json")

    model = build_model(spec, coverage=label_coverage(data, train, spec))   # same initial weights as mffer.train
    log = RoundLog(run_dir, model, make_dataset(data, val, spec, stats, training=False), config)
    if log.done:
        model.load_weights(log.last)
        print(f"resuming after round {log.done}")
    else:
        meta.update(parameters=model.count_params(), update_bytes=4 * model.count_params(),
                    clients=[{"clips": len(c), "actors": int(data.clips.actor_id.iloc[c].nunique()),
                              **({"valence_class_weights": valence_class_weights(data, c)} if spec.balance_valence
                                 else {})} for c in clients],
                    client_optimizer="fresh Adam(1e-4) every round; clients keep no state")
    meta["client_devices_checked"] = True   # aggregate_fit raises if a client used another device
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))

    if not log.finished:
        start = time.perf_counter()
        strategy = make_strategy(log, len(clients), model.get_weights(), meta["device"])
        rounds = config["max_epochs"] - log.done
        run_simulation(
            server_app=ServerApp(server_fn=lambda context: ServerAppComponents(
                strategy=strategy, config=ServerConfig(num_rounds=rounds))),
            client_app=make_client_app(run_dir), num_supernodes=len(clients),
            # One client at a time: each asks for all CPUs (and the GPU on CUDA, which Ray then exposes to it).
            backend_config={"client_resources": {"num_cpus": os.cpu_count(),
                                                 "num_gpus": 1.0 if meta["device"] == "cuda" else 0.0}},
        )
        meta["train_minutes"] = meta.get("train_minutes", 0) + (time.perf_counter() - start) / 60
        # Flower may log a server error instead of raising it: check the outcome before scoring test.
        log = RoundLog(run_dir, model, log.val, config)
        if not log.finished or not (run_dir / "best.weights.h5").exists():
            raise SystemExit(f"simulation ended after {log.done} of {config['max_epochs']} rounds without "
                             f"early stopping: see the Flower log above; continue with --resume {run_dir}")

    if not config.get("score_test", True):
        return score_val(run_dir, log.best_round, meta)
    return score_test(run_dir, model, data, test, spec, stats, log.best_round, meta)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", type=Path)
    parser.add_argument("--split", type=int)
    parser.add_argument("--seed", type=int, help="weight-initialisation seed (default: the split id)")
    parser.add_argument("--resume", type=Path, help="run directory to continue")
    args = parser.parse_args()
    print(f"device: {device.configure()} {device.names()}")
    if args.resume:
        run_dir = args.resume.resolve()
    else:
        assert args.config and args.split is not None, "give a config and --split, or --resume"
        assert "federated" in {**DEFAULTS, **load_run_config(args.config)}, "config has no `federated:` section"
        run_dir = new_run(args.config, args.split, args.seed)
    print(f"run directory: {run_dir}")
    print_result(run(run_dir))


if __name__ == "__main__":
    # Import from the package, not __main__, so Flower's workers unpickle these functions by reference.
    from mffer.federated import main as package_main
    package_main()
