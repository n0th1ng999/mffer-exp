"""Centralised training of one configuration on one split.

    python -m mffer.train configs/runs/c1_m1.yaml --split 0
    python -m mffer.train configs/runs/l1_local.yaml --split 0 --client 1   # local-only: one client's rows
    python -m mffer.train --resume runs/c1_m1-s0-20261118-2030      # continue after a crash

Same protocol for every configuration: Adam 1e-4, batch 16, at most `max_epochs`, early stopping with
`patience` on `monitor` (val valence macro-F1, the primary metric: ADR 0014), never before `min_epochs` (ADR 0023). The full model is saved after every epoch (so a
crash costs one epoch) and the best epoch's weights separately. The test split is scored exactly once, at
the end, with the best weights; its predictions and metrics are written to the run directory.

Local-only mode (L1): a config with `local: {clients: N, partition: actor}` plus `--client k` trains on the
training rows of federated client k only (`splits.client_rows`, the same rows that client holds in F2) and
is otherwise identical: same val and test actors, same stopping rule. Run name: `<name>_c<k>`.

Device: CPU on macOS (the Metal GPU diverges), CUDA on Linux/WSL2 with an NVIDIA GPU; see mffer.device and
ADR 0015. `MFFER_CPU=1` forces the CPU.
"""
import argparse
import json
import os
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

from mffer import device
from mffer.config import ROOT, load_paths, load_run_config
from mffer.data.cremad import VALENCES
from mffer.data.pipeline import AudioStats, Cremad, fit_stats, label_coverage, make_dataset, valence_class_weights
from mffer.data.splits import client_rows, rows
from mffer.metrics import MONITOR, SCORE, compute, predictions_table, score
from mffer.models import build_model, derive_seed, reseed, spec_from_config

# protocol: 4 since 25 Nov 2026 (ADR 0026: monitor val_score, the mean of valence and emotion macro-F1 per corpus;
# ADR 0023 still sets min_epochs 20, patience 10, max_epochs 150); finished runs of an older protocol are
# "invalid (old protocol)". score_test: false for dev runs (configs/dev): they stop at validation and never
# touch the test actors. dataset: "cremad" (V1) or "v2" (CREMA-D + RAVDESS, ADR 0024).
PROTOCOL = 4
DEFAULTS = {"max_epochs": 150, "min_epochs": 20, "patience": 10, "monitor": SCORE, "batch_size": 16,
            "protocol": PROTOCOL, "score_test": True, "dataset": "cremad"}


def should_stop(done: int, since_best: int, config: dict) -> bool:
    """The stopping rule of every run, centralised or federated (ADR 0023): the epoch/round budget is spent, or
    at least `min_epochs` are done and the monitor has not improved for `patience` of them. The minimum keeps
    early stopping from ending a run while valence still predicts the majority class (flat monitor)."""
    return done >= config["max_epochs"] or (done >= config.get("min_epochs", 0) and since_best >= config["patience"])


def val_scorer(data: Cremad, val: np.ndarray, spec, stats: AudioStats):
    """`scorer(model)` -> the validation selection score and its per-corpus parts, as `val_*` log entries."""
    dataset = make_dataset(data, val, spec, stats, training=False)

    def scorer(model: keras.Model) -> dict[str, float]:
        table = predictions_table(data.clips, val, data.targets, data.weights, model.predict(dataset, verbose=0))
        return {f"val_{key}": value for key, value in score(table).items()}
    return scorer


class ValScore(keras.callbacks.Callback):
    """Adds `val_score` (protocol 4's monitor) and its parts to the epoch's logs. First in the callback list, so
    CSVLogger writes them and EpochCheckpoint selects on them."""

    def __init__(self, scorer):
        super().__init__()
        self.scorer = scorer

    def on_epoch_end(self, epoch, logs=None):
        logs.update(self.scorer(self.model))


class Reseed(keras.callbacks.Callback):
    """Dropout state from (seed, epoch) at the start of every epoch, so a resumed run trains exactly as an
    uninterrupted one (ADR 0016). `epoch` is absolute: Keras counts from `initial_epoch`."""

    def __init__(self, seed: int):
        super().__init__()
        self.seed = seed

    def on_epoch_begin(self, epoch, logs=None):
        reseed(self.model, derive_seed(self.seed, epoch))


def save_atomic(path: Path, save) -> None:
    """`save(tmp)` then rename: a crash mid-save never leaves a half-written file under `path`."""
    tmp = path.with_name(f"tmp-{path.name}")
    save(tmp)
    os.replace(tmp, path)


def save_checkpoint(run_dir: Path, done: int, suffix: str, save) -> None:
    """The resume point after `done` epochs (or rounds), named `last-<done><suffix>` so the file itself says how
    far it is; older resume points are removed after the new one is complete."""
    path = run_dir / f"last-{done:03d}{suffix}"
    save_atomic(path, save)
    for old in run_dir.glob(f"last-*{suffix}"):
        if old != path:
            old.unlink()


def load_history(run_dir: Path, suffix: str) -> tuple[pd.DataFrame, Path | None]:
    """epochs.csv cut back to the latest resume point, and that point (None before the first one).

    Each epoch appends its row first and saves its resume point second. A crash between the two leaves one
    row more than the resume point holds; that row is dropped and the epoch runs again, which reproduces it
    exactly (ADR 0016)."""
    log_file = run_dir / "epochs.csv"
    history = pd.read_csv(log_file) if log_file.exists() else pd.DataFrame()
    points = sorted(run_dir.glob(f"last-*{suffix}"))
    done = int(points[-1].name.removeprefix("last-").removesuffix(suffix)) if points else 0
    assert len(history) >= done, f"{run_dir}: resume point after {done} epochs but epochs.csv has {len(history)}"
    if len(history) > done:
        print(f"epochs.csv has {len(history)} rows, the resume point {done}: dropping the rest and redoing them")
        history = history.iloc[:done]
        history.to_csv(log_file, index=False)
    return history, points[-1] if points else None


class EpochCheckpoint(keras.callbacks.Callback):
    """After each epoch: keep the best weights, save the full model (resume point), stop by `should_stop`.
    Runs after CSVLogger, so the epoch's row is written before its resume point
    (see `load_history`). State is rebuilt from epochs.csv on resume."""

    def __init__(self, run_dir: Path, config: dict, history: pd.DataFrame):
        super().__init__()
        self.run_dir, self.config, monitor = run_dir, config, config["monitor"]
        self.monitor, self.done = monitor, len(history)
        self.best = history[monitor].max() if len(history) else -np.inf
        self.best_epoch = int(history[monitor].idxmax()) if len(history) else -1
        self.since_best = len(history) - 1 - self.best_epoch if len(history) else 0

    def on_epoch_end(self, epoch, logs=None):
        value = logs[self.monitor]
        if value > self.best:
            self.best, self.best_epoch, self.since_best = value, epoch, 0
            save_atomic(self.run_dir / "best.weights.h5", self.model.save_weights)
        else:
            self.since_best += 1
        self.done = epoch + 1
        save_checkpoint(self.run_dir, self.done, ".keras", self.model.save)
        if should_stop(self.done, self.since_best, self.config):
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
    if seed is not None and seed != split_id:   # seed-variance runs get their own config name (and directory)
        config["name"] = f"{config['name']}_seed{seed}"
    run_dir = load_paths().runs / f"{config['name']}-s{split_id}-{datetime.now():%Y%m%d-%H%M}"
    run_dir.mkdir(parents=True)
    meta = {"config_file": str(config_file), "split": split_id, "seed": split_id if seed is None else seed,
            "git": git_version(), "tensorflow": tf.__version__, "keras": keras.__version__,
            "machine": f"{platform.platform()} {platform.processor()}",
            "device": device.current(), "gpu": device.names(), "deterministic": device.deterministic(),
            "started": datetime.now().isoformat()}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return run_dir


def run(run_dir: Path) -> dict:
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    meta = json.loads((run_dir / "meta.json").read_text())
    device.check_resume(meta)
    keras.utils.set_random_seed(meta["seed"])
    spec = spec_from_config(config["model"])
    dataset = config.get("dataset", "cremad")
    data = Cremad(dataset=dataset)
    train, val, test = (rows(data.clips, meta["split"], p, dataset) for p in ("train", "val", "test"))
    if "local" in config:
        local = config["local"]
        train = client_rows(data.clips, meta["split"], local["clients"], local["partition"], dataset)[local["client"]]

    stats_file = run_dir / "audio_stats.json"
    if not stats_file.exists():
        fit_stats(data, train, spec.audio_features).save(stats_file)
    stats = AudioStats.load(stats_file)
    coverage = label_coverage(data, train, spec)

    log_file = run_dir / "epochs.csv"
    history, last = load_history(run_dir, ".keras")
    if last:
        model = keras.models.load_model(last)
        print(f"resuming after epoch {len(history)}")
    else:
        model = build_model(spec, coverage=coverage)
        meta.update(parameters=model.count_params(), model_bytes=4 * model.count_params(),
                    trainable_parameters=int(sum(np.prod(w.shape) for w in model.trainable_weights)),
                    train_clips=len(train), coverage=coverage,
                    loss_weights_applied={h: w / coverage[h] for h, w in spec.loss_weights.items()})
        if spec.balance_valence:
            meta["valence_class_weights"] = dict(zip(VALENCES, valence_class_weights(data, train), strict=True))
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))

    checkpoint = EpochCheckpoint(run_dir, config, history)
    if not should_stop(checkpoint.done, checkpoint.since_best, config):
        start = time.perf_counter()
        model.fit(make_dataset(data, train, spec, stats, training=True, seed=meta["seed"],
                               batch_size=config["batch_size"], first_epoch=len(history)),
                  validation_data=make_dataset(data, val, spec, stats, training=False),
                  epochs=config["max_epochs"], initial_epoch=len(history), verbose=2, shuffle=False,  # pipeline shuffles
                  callbacks=[ValScore(val_scorer(data, val, spec, stats)), Reseed(meta["seed"]),
                             keras.callbacks.CSVLogger(log_file, append=True), checkpoint])
        meta["train_minutes"] = meta.get("train_minutes", 0) + (time.perf_counter() - start) / 60

    if not config.get("score_test", True):
        return score_val(run_dir, checkpoint.best_epoch, meta)
    return score_test(run_dir, model, data, test, spec, stats, checkpoint.best_epoch, meta)


def score_val(run_dir: Path, best_epoch: int, meta: dict) -> dict:
    """Dev runs (score_test: false): the validation scores at the best epoch, in val.json; test is not read."""
    history = pd.read_csv(run_dir / "epochs.csv")
    result = {"best_epoch": int(best_epoch) + 1, "epochs_run": len(history),
              "val_at_best": history.iloc[best_epoch].to_dict()}
    (run_dir / "val.json").write_text(json.dumps(result, indent=1))
    meta["finished"] = datetime.now().isoformat()
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return result


def print_result(result: dict) -> None:
    if "test" in result:
        return print_test(result["test"])
    val = result["val_at_best"]
    print(f"dev run, best epoch {result['best_epoch']} of {result['epochs_run']} (test not scored):")
    for key in (SCORE, MONITOR, "val_displayed_intensity_mae", "val_perceived_intensity_mae",
                "val_displayed_emotion_macro_f1", "val_perceived_emotion_macro_f1"):
        if key in val:
            print(f"  {key}: {val[key]:.4f}")


def score_test(run_dir: Path, model: keras.Model, data: Cremad, test: np.ndarray, spec, stats: AudioStats,
               best_epoch: int, meta: dict) -> dict:
    """Score the test rows once, with the best epoch's (or round's) weights; write predictions.csv,
    test.json and the finish time. `best_epoch` is 0-based, as in epochs.csv."""
    model.load_weights(run_dir / "best.weights.h5")
    start = time.perf_counter()
    outputs = model.predict(make_dataset(data, test, spec, stats, training=False), verbose=0)
    meta["inference_ms_per_clip"] = 1000 * (time.perf_counter() - start) / len(test)   # incl. reading inputs (T6)
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
    """Primary metric first, then the intensities (ADR 0014), then the other heads."""
    for head in ("displayed_valence", "displayed_intensity", "perceived_intensity",
                 "displayed_emotion", "perceived_emotion"):
        if head not in test:
            continue
        if "mae" in test[head]:
            print(f"test {head}: MAE {test[head]['mae']:.3f}")
        else:
            print(f"test {head}: macro-F1 {test[head]['macro_f1']:.3f}, accuracy {test[head]['accuracy']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", type=Path)
    parser.add_argument("--split", type=int)
    parser.add_argument("--seed", type=int, help="weight-initialisation seed (default: the split id)")
    parser.add_argument("--client", type=int, help="local-only: train on this federated client's rows only")
    parser.add_argument("--resume", type=Path, help="run directory to continue")
    args = parser.parse_args()
    print(f"device: {device.configure()} {device.names()}")
    if args.resume:
        run_dir = args.resume
    else:
        assert args.config and args.split is not None, "give a config and --split, or --resume"
        run_dir = new_run(args.config, args.split, args.seed, args.client)
    print(f"run directory: {run_dir}")
    print_result(run(run_dir))


if __name__ == "__main__":
    main()
