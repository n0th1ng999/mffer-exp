"""Resume points (train.load_history): a crash between an epoch's row and its resume point drops the row,
so the epoch runs again instead of being skipped."""
import pandas as pd

from mffer.train import load_history, save_checkpoint


def write_rows(run_dir, n):
    pd.DataFrame({"epoch": range(n), "val_x": range(n)}).to_csv(run_dir / "epochs.csv", index=False)


def touch(path):
    path.write_text("weights")


def test_no_resume_point_yet_means_a_fresh_run(tmp_path):
    write_rows(tmp_path, 1)                               # crashed after epoch 1's row, before its checkpoint
    history, last = load_history(tmp_path, ".keras")
    assert last is None and len(history) == 0
    assert len(pd.read_csv(tmp_path / "epochs.csv")) == 0


def test_a_row_without_its_resume_point_is_dropped(tmp_path):
    write_rows(tmp_path, 3)
    save_checkpoint(tmp_path, 2, ".keras", touch)
    history, last = load_history(tmp_path, ".keras")
    assert last.name == "last-002.keras" and len(history) == 2
    assert len(pd.read_csv(tmp_path / "epochs.csv")) == 2   # the file is cut back too


def test_matching_rows_and_resume_point_are_kept(tmp_path):
    write_rows(tmp_path, 3)
    save_checkpoint(tmp_path, 3, ".weights.h5", touch)
    history, last = load_history(tmp_path, ".weights.h5")
    assert last.name == "last-003.weights.h5" and len(history) == 3


def test_only_the_latest_resume_point_is_kept(tmp_path):
    for done in (1, 2, 3):
        save_checkpoint(tmp_path, done, ".keras", touch)
    assert [p.name for p in tmp_path.iterdir()] == ["last-003.keras"]


# ---------------------------------------------------------------- stopping rule (ADR 0023)

def test_no_early_stop_before_min_epochs_even_with_a_flat_monitor():
    from mffer.train import DEFAULTS, should_stop
    assert not should_stop(done=6, since_best=5, config=DEFAULTS)                       # the old patience-5 stop
    assert not should_stop(done=19, since_best=18, config=DEFAULTS)                     # flat since epoch 1
    assert should_stop(done=20, since_best=19, config=DEFAULTS)
    assert not should_stop(done=30, since_best=9, config=DEFAULTS)
    assert should_stop(done=30, since_best=10, config=DEFAULTS)
    assert should_stop(done=90, since_best=0, config=DEFAULTS)                          # budget spent
