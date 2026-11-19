"""B0: the floor every model must beat. No learning, no inputs read.

    python -m mffer.baselines            # all three splits

Per split, from the training rows only: each classification head predicts its most frequent training
class for every test clip (majority class); each intensity head predicts the training mean (constant
predictor). Masked labels are ignored, as in training. Writes a normal run directory
(`b0_majority-s<k>-<stamp>/`: config, meta, predictions.csv, test.json), scored by the same
`mffer.metrics` code as every model, so the reporter treats it like any other run.
"""
import json
from datetime import datetime

import numpy as np
import yaml

from mffer.config import load_paths
from mffer.data.pipeline import Cremad
from mffer.data.splits import N_SPLITS, rows
from mffer.metrics import compute, predictions_table
from mffer.models import HEADS
from mffer.train import git_version

NAME = "b0_majority"


def fit(data: Cremad, train: np.ndarray) -> dict[str, np.ndarray]:
    """One output vector per head, shaped like the model's: one-hot of the majority class, or the mean."""
    out = {}
    for head, (units, _, loss) in HEADS.items():
        present = data.weights[head][train] == 1
        targets = data.targets[head][train][present]
        if loss == "mse":
            out[head] = np.array([targets[:, 0].mean()], np.float32)
        else:
            out[head] = np.eye(units, dtype=np.float32)[targets.sum(0).argmax()]
    return out


def run_split(data: Cremad, split_id: int) -> dict:
    train, test = rows(data.clips, split_id, "train"), rows(data.clips, split_id, "test")
    constant = fit(data, train)
    run_dir = load_paths().runs / f"{NAME}-s{split_id}-{datetime.now():%Y%m%d-%H%M}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"name": NAME, "model": "majority class / train mean"}))
    meta = {"split": split_id, "git": git_version(), "gpu": [], "parameters": 0,
            "predicts": {h: v.tolist() for h, v in constant.items()}, "started": datetime.now().isoformat()}
    outputs = {h: np.tile(v, (len(test), 1)) for h, v in constant.items()}
    table = predictions_table(data.clips, test, data.targets, data.weights, outputs)
    table.to_csv(run_dir / "predictions.csv", index=False)
    result = {"best_epoch": None, "epochs_run": 0, "test": compute(table)}
    (run_dir / "test.json").write_text(json.dumps(result, indent=1))
    meta["finished"] = datetime.now().isoformat()
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return result["test"]


def main() -> None:
    data = Cremad()
    for split_id in range(N_SPLITS):
        test = run_split(data, split_id)
        print(f"split {split_id}: displayed {test['displayed_emotion']['accuracy']:.3f}, "
              f"perceived {test['perceived_emotion']['accuracy']:.3f}, "
              f"valence {test['displayed_valence']['accuracy']:.3f}, "
              f"perceived-intensity MAE {test['perceived_intensity']['mae']:.3f}")


if __name__ == "__main__":
    main()
