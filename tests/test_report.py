"""Which runs the reporter counts and how it aggregates them (fake run directories, no training)."""
import json

import pytest

from mffer.report import load_runs, per_split, results_table


def fake_run(root, name, accuracy, gpu=(), finished=True):
    path = root / name
    path.mkdir()
    (path / "meta.json").write_text(json.dumps({"gpu": list(gpu), "git": "abc", "parameters": 10,
                                                "update_bytes": 40, "train_minutes": 6.0}))
    if finished:
        (path / "test.json").write_text(json.dumps({"best_epoch": 2, "epochs_run": 3, "test": {
            "displayed_emotion": {"accuracy": accuracy, "accuracy_by_sex": {"Female": accuracy, "Male": accuracy}}}}))


@pytest.fixture
def runs(tmp_path):
    fake_run(tmp_path, "c2_m3-s0-20261120-1000", 0.40)
    fake_run(tmp_path, "c2_m3-s0-20261121-1000", 0.50)                  # later: supersedes the first
    fake_run(tmp_path, "c2_m3-s0-20261122-1000", 0.90, gpu=["gpu:0"])   # GPU: never counted
    fake_run(tmp_path, "c2_m3-s1-20261122-1100", 0.60, finished=False)  # incomplete
    fake_run(tmp_path, "c2_m3-s1-20261121-1100", 0.70)
    fake_run(tmp_path, "l1_local_c0-s0-20261123-1000", 0.30)
    fake_run(tmp_path, "l1_local_c1-s0-20261123-1100", 0.40)
    fake_run(tmp_path, "l1_local_c0-s1-20261123-1200", 0.30)           # client 1 missing on split 1
    return load_runs(tmp_path)


def test_latest_finished_cpu_run_counts(runs):
    status = dict(zip(runs.run, runs.status))
    assert status["c2_m3-s0-20261121-1000"] == "counted"
    assert status["c2_m3-s0-20261120-1000"] == "superseded"
    assert status["c2_m3-s0-20261122-1000"] == "invalid (GPU)"
    assert status["c2_m3-s1-20261122-1100"] == "incomplete"
    assert status["c2_m3-s1-20261121-1100"] == "counted"


def test_local_only_averages_both_clients_and_needs_both(runs):
    scores = per_split(runs, "L1")
    assert list(scores.index) == [0]
    assert scores.loc[0, "displayed_emotion.accuracy"] == pytest.approx(0.35)


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
