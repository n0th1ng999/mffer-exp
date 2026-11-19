"""Centralised training of one configuration on one split.

    python -m mffer.train configs/runs/c1_m1.yaml --split 0
    python -m mffer.train configs/runs/l1_local.yaml --split 0 --client 1   # local-only: one client's rows
    python -m mffer.train --resume runs/c1_m1-s0-20261118-2030      # continue after a crash

Same protocol for every configuration: Adam 1e-4, batch 16, at most `max_epochs`, early stopping with
`patience` on `monitor` (val displayed-emotion accuracy). The full model is saved after every epoch (so a
crash costs one epoch) and the best epoch's weights separately. The test split is scored exactly once, at
the end, with the best weights; its predictions and metrics are written to the run directory.

Local-only mode (L1): a config with `local: {clients: N, partition: actor}` plus `--client k` trains on the
training rows of federated client k only (`splits.client_rows`, the same rows that client holds in F2) and
is otherwise identical: same val and test actors, same stopping rule. Run name: `<name>_c<k>`.

Training runs on the CPU: on the Metal GPU (tensorflow-metal 1.2.0, with TF 2.18 and 2.19) the same code,
data and seed diverge after 3-4 epochs while the CPU trains cleanly (scripts/lab_stability.py, 18 Nov 2026).
"""
import argparse
import json
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from mffer.config import ROOT, load_paths, load_run_config
from mffer.data.pipeline import AudioStats, Cremad, fit_stats, label_coverage, make_dataset
from mffer.data.splits import client_rows, rows
from mffer.metrics import compute, predictions_table
from mffer.models import build_model, spec_from_config

DEFAULTS = {"max_epochs": 90, "patience": 5, "monitor": "val_displayed_emotion_accuracy", "batch_size": 16}


class EpochCheckpoint(keras.callbacks.Callback):
    """After each epoch: save the full model (resume point), keep the best weights, stop after `patience`
    epochs without improvement. State is rebuilt from epochs.csv on resume."""

    def __init__(self, run_dir: Path, monitor: str, patience: int, history: pd.DataFrame):
        super().__init__()
        self.run_dir, self.monitor, self.patience = run_dir, monitor, patience
        self.best = history[monitor].max() if len(history) else -np.inf
        self.best_epoch = int(history[monitor].idxmax()) if len(history) else -1
        self.since_best = len(history) - 1 - self.best_epoch if len(history) else 0

    def on_epoch_end(self, epoch, logs=None):
        value = logs[self.monitor]
        if value > self.best:
            self.best, self.best_epoch, self.since_best = value, epoch, 0
            self.model.save_weights(self.run_dir / "best.weights.h5")
        else:
            self.since_best += 1
        self.model.save(self.run_dir / "last.keras")
        if self.since_best >= self.patience:
            self.model.stop_training = True


def git_version() -> str:
    try:
        return subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        return "no-commit"


def new_run(config_file: Path, split_id: int, seed: int | None, client: int | None = None) -> Path:
    config = {**DEFAULTS, **load_run_config(config_file)}
    if client is not None:
        assert "local" in config, "--client needs a config with a `local:` section"
        assert 0 <= client < config["local"]["clients"], f"client must be < {config['local']['clients']}"
        config["local"] = {**config["local"], "client": client}
        config["name"] = f"{config['name']}_c{client}"
    else:
        assert "local" not in config, "a local-only config needs --client"
    run_dir = load_paths().runs / f"{config['name']}-s{split_id}-{datetime.now():%Y%m%d-%H%M}"
    run_dir.mkdir(parents=True)
    meta = {"config_file": str(config_file), "split": split_id, "seed": split_id if seed is None else seed,
            "git": git_version(), "tensorflow": tf.__version__, "keras": keras.__version__,
            "machine": f"{platform.platform()} {platform.processor()}",
            "gpu": [d.name for d in tf.config.get_visible_devices("GPU")],   # [] = trained on the CPU
             "started": datetime.now().isoformat()}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return run_dir


def run(run_dir: Path) -> dict:
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    meta = json.loads((run_dir / "meta.json").read_text())
    keras.utils.set_random_seed(meta["seed"])
    spec = spec_from_config(config["model"])
    data = Cremad()
    train, val, test = (rows(data.clips, meta["split"], p) for p in ("train", "val", "test"))
    if "local" in config:
        local = config["local"]
        train = client_rows(data.clips, meta["split"], local["clients"], local["partition"])[local["client"]]

    stats_file = run_dir / "audio_stats.json"
    if not stats_file.exists():
        fit_stats(data, train).save(stats_file)
    stats = AudioStats.load(stats_file)
    coverage = label_coverage(data, train, spec)

    log_file = run_dir / "epochs.csv"
    history = pd.read_csv(log_file) if log_file.exists() else pd.DataFrame()
    if len(history):
        model = keras.models.load_model(run_dir / "last.keras")
        print(f"resuming after epoch {len(history)}")
    else:
        model = build_model(spec, coverage=coverage)
        meta.update(parameters=model.count_params(), coverage=coverage,
                    loss_weights_applied={h: w / coverage[h] for h, w in spec.loss_weights.items()})
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))

    checkpoint = EpochCheckpoint(run_dir, config["monitor"], config["patience"], history)
    done = len(history) >= config["max_epochs"] or checkpoint.since_best >= config["patience"]
    if not done:
        start = time.perf_counter()
        model.fit(make_dataset(data, train, spec, stats, training=True, seed=meta["seed"],
                               batch_size=config["batch_size"]),
                  validation_data=make_dataset(data, val, spec, stats, training=False),
                  epochs=config["max_epochs"], initial_epoch=len(history), verbose=2, shuffle=False,  # pipeline shuffles
                  callbacks=[keras.callbacks.CSVLogger(log_file, append=True), checkpoint])
        meta["train_minutes"] = meta.get("train_minutes", 0) + (time.perf_counter() - start) / 60

    return score_test(run_dir, model, data, test, spec, stats, checkpoint.best_epoch, meta)


def score_test(run_dir: Path, model: keras.Model, data: Cremad, test: np.ndarray, spec, stats: AudioStats,
               best_epoch: int, meta: dict) -> dict:
    """Score the test rows once, with the best epoch's (or round's) weights; write predictions.csv,
    test.json and the finish time. `best_epoch` is 0-based, as in epochs.csv."""
    model.load_weights(run_dir / "best.weights.h5")
    outputs = model.predict(make_dataset(data, test, spec, stats, training=False), verbose=0)
    table = predictions_table(data.clips, test, data.targets, data.weights, outputs)
    table.to_csv(run_dir / "predictions.csv", index=False)
    history = pd.read_csv(run_dir / "epochs.csv")
    result = {"best_epoch": int(best_epoch) + 1, "epochs_run": len(history),
              "val_at_best": history.iloc[best_epoch].to_dict(), "test": compute(table)}
    (run_dir / "test.json").write_text(json.dumps(result, indent=1))
    meta["finished"] = datetime.now().isoformat()
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return result


def print_test(test: dict) -> None:
    for head in ("displayed_emotion", "perceived_emotion", "displayed_valence"):
        if head in test:
            print(f"test {head}: accuracy {test[head]['accuracy']:.3f}, macro-F1 {test[head]['macro_f1']:.3f}")
    for head in ("displayed_intensity", "perceived_intensity"):
        if head in test:
            print(f"test {head}: MAE {test[head]['mae']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", type=Path)
    parser.add_argument("--split", type=int)
    parser.add_argument("--seed", type=int, help="weight-initialisation seed (default: the split id)")
    parser.add_argument("--client", type=int, help="local-only: train on this federated client's rows only")
    parser.add_argument("--resume", type=Path, help="run directory to continue")
    args = parser.parse_args()
    tf.config.set_visible_devices([], "GPU")   # CPU only: see the module docstring
    if args.resume:
        run_dir = args.resume
    else:
        assert args.config and args.split is not None, "give a config and --split, or --resume"
        run_dir = new_run(args.config, args.split, args.seed, args.client)
    print(f"run directory: {run_dir}")
    print_test(run(run_dir)["test"])


if __name__ == "__main__":
    main()
