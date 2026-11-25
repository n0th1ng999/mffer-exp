"""Which runs the reporter counts and how it aggregates them (fake run directories, no training)."""
import json

import pytest
import yaml

from mffer.metrics import SCORE as MONITOR
from mffer.report import load_runs, per_split, results_table
from mffer.train import PROTOCOL


def fake_run(root, name, score, gpu=(), finished=True, monitor=MONITOR, device=None, protocol=PROTOCOL, dev=False):
    """`score` is the primary metric, valence macro-F1. `device` None = a run from before mffer.device."""
    path = root / name
    path.mkdir()
    meta = {"gpu": list(gpu), "git": "abc", "parameters": 10, "update_bytes": 40, "train_minutes": 6.0}
    (path / "meta.json").write_text(json.dumps(meta if device is None else {**meta, "device": device}))
    config = {"name": name.split("-")[0], "monitor": monitor, "model": {"preset": "m3"}}
    (path / "config.yaml").write_text(yaml.safe_dump(config if protocol is None else {**config, "protocol": protocol}))
    if dev:
        (path / "val.json").write_text(json.dumps({"best_epoch": 2, "epochs_run": 3, "val_at_best": {}}))
    elif finished:
        (path / "test.json").write_text(json.dumps({"best_epoch": 2, "epochs_run": 3, "test": {
            "displayed_valence": {"macro_f1": score, "macro_f1_by_sex": {"Female": score, "Male": score}}}}))


@pytest.fixture
def runs(tmp_path):
    fake_run(tmp_path, "c2_m3-s0-20261120-1000", 0.40)
    fake_run(tmp_path, "c2_m3-s0-20261121-1000", 0.50)                  # later: supersedes the first
    fake_run(tmp_path, "c2_m3-s0-20261122-1000", 0.90, gpu=["gpu:0"])   # old run on a GPU = Metal: never counted
    fake_run(tmp_path, "u1_audio-s0-20261124-1000", 0.55, gpu=["NVIDIA GeForce RTX 4080"], device="cuda")
    fake_run(tmp_path, "u1_audio-s1-20261124-1000", 0.95, gpu=["gpu:0"], device="metal")
    fake_run(tmp_path, "c2_m3-s1-20261122-1100", 0.60, finished=False)  # incomplete
    fake_run(tmp_path, "c2_m3-s1-20261121-1100", 0.70)
    fake_run(tmp_path, "l1_local_c0-s0-20261123-1000", 0.30)
    fake_run(tmp_path, "l1_local_c1-s0-20261123-1100", 0.40)
    fake_run(tmp_path, "l1_local_c0-s1-20261123-1200", 0.30)           # client 1 missing on split 1
    fake_run(tmp_path, "c2_m3-s2-20261123-1300", 0.80, monitor="val_displayed_emotion_accuracy")  # old rule
    fake_run(tmp_path, "c2_m3-s2-20261124-0900", 0.85, protocol=None)   # before ADR 0017: protocol 1
    fake_run(tmp_path, "c2_m3-s2-20261124-1000", 0.95, dev=True)        # dev run: val only, never counts
    return load_runs(tmp_path)


def test_old_protocol_and_dev_runs_never_count(runs):
    status = dict(zip(runs.run, runs.status, strict=True))
    assert status["c2_m3-s2-20261124-0900"] == "invalid (old protocol)"
    assert status["c2_m3-s2-20261124-1000"] == "dev (val only)"
    assert per_split(runs, "C2").index.tolist() == [0, 1]   # split 2 has no counted run


def test_latest_finished_cpu_run_counts(runs):
    status = dict(zip(runs.run, runs.status, strict=True))
    assert status["c2_m3-s0-20261121-1000"] == "counted"
    assert status["c2_m3-s0-20261120-1000"] == "superseded"
    assert status["c2_m3-s0-20261122-1000"] == "invalid (GPU)"
    assert status["c2_m3-s1-20261122-1100"] == "incomplete"
    assert status["c2_m3-s1-20261121-1100"] == "counted"
    assert status["c2_m3-s2-20261123-1300"] == "invalid (old monitor)"
    assert status["u1_audio-s0-20261124-1000"] == "counted"          # CUDA counts (ADR 0015)
    assert status["u1_audio-s1-20261124-1000"] == "invalid (GPU)"   # Metal never does   # epoch picked by another metric


def test_local_only_averages_both_clients_and_needs_both(runs):
    scores = per_split(runs, "L1")
    assert list(scores.index) == [0]
    assert scores.loc[0, "displayed_valence.macro_f1"] == pytest.approx(0.35)


def test_federated_table_gap_to_c2_is_paired_by_split(runs):
    table = results_table(runs, ["C2", "L1"], federated=True)
    l1 = next(line for line in table.splitlines() if line.startswith("L1 "))
    assert l1.rstrip(" \\").endswith("-15.0")          # 35% - 50% on split 0, the only shared split
    assert "c2_m3-s0-20261121-1000" in table and "c2_m3-s0-20261120-1000" not in table


def test_restricting_to_split_0_drops_the_other_splits(runs):
    assert list(per_split(runs, "C2").index) == [0, 1]
    assert list(per_split(runs, "C2", [0]).index) == [0]
    table = results_table(runs, ["C2"], splits=[0])
    assert "c2_m3-s1-20261121-1100" not in table and "split 0 only" in table
