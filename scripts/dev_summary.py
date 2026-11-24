"""Summary of the dev sweep (ADR 0017): validation scores of every finished dev run, mean ± sd over seeds.

    python scripts/dev_summary.py            # also written to runs/dev_summary.md

Rows follow the ladder order of configs/dev (dev_<model>_<step>_<change>). "Δ" is the change in mean val
valence macro-F1 against the previous step of the same ladder (steps 6+ against step 5, the branch point);
a step is worth keeping when Δ is clearly larger than the seed sd of both steps. Test is never read.
"""
import json
import re
import statistics

import yaml

from mffer.config import load_paths
from mffer.metrics import MONITOR

COLUMNS = [(MONITOR, "val valence F1", 100, 1), ("val_displayed_intensity_mae", "val displ. int. MAE", 1, 3),
           ("val_perceived_intensity_mae", "val perc. int. MAE", 1, 3),
           ("val_displayed_emotion_macro_f1", "val displ. emo. F1", 100, 1)]
NAME = re.compile(r"^(?P<config>dev_(?P<model>[a-z0-9]+)_(?P<step>\d+)_[a-z0-9]+?)(_seed(?P<seed>\d+))?-s0-\d{8}-\d{4}$")


def load() -> dict:
    runs: dict = {}
    for path in sorted(load_paths().runs.glob("dev_*")):
        match = NAME.match(path.name)
        if not match or not (path / "val.json").exists():
            continue
        result = json.loads((path / "val.json").read_text())
        meta = json.loads((path / "meta.json").read_text())
        entry = runs.setdefault(match["config"], {"model": match["model"], "step": int(match["step"]), "seeds": {}})
        entry["seeds"][int(match["seed"] or 0)] = {
            **result["val_at_best"], "best_epoch": result["best_epoch"], "params": meta.get("parameters"),
            "min_per_epoch": meta.get("train_minutes", 0) / max(result["epochs_run"], 1)}
        entry["flags"] = yaml.safe_load((path / "config.yaml").read_text())["model"]
    return runs


def mean_sd(values: list[float], scale: float, digits: int) -> str:
    if len(values) == 1:
        return f"{scale * values[0]:.{digits}f}"
    return f"{scale * statistics.mean(values):.{digits}f} ± {scale * statistics.stdev(values):.{digits}f}"


def main() -> None:
    runs = load()
    lines = ["| Config | Seeds | " + " | ".join(h for _, h, _, _ in COLUMNS) + " | Δ F1 | Best epoch | Params | min/epoch |",
             "|---" * (len(COLUMNS) + 6) + "|"]
    for config, entry in sorted(runs.items(), key=lambda kv: (kv[1]["model"], kv[1]["step"])):
        seeds = list(entry["seeds"].values())
        cells = [mean_sd([s[key] for s in seeds if key in s], scale, digits) if any(key in s for s in seeds) else "–"
                 for key, _, scale, digits in COLUMNS]
        previous_step = 5 if entry["step"] > 5 else entry["step"] - 1
        previous = next((e for e in runs.values() if e["model"] == entry["model"] and e["step"] == previous_step), None)
        delta = "–"
        if previous:
            now = statistics.mean(s[MONITOR] for s in seeds)
            before = statistics.mean(s[MONITOR] for s in previous["seeds"].values())
            delta = f"{100 * (now - before):+.1f}"
        lines.append(f"| {config} | {len(seeds)} | " + " | ".join(cells) + f" | {delta} | "
                     f"{mean_sd([s['best_epoch'] for s in seeds], 1, 0)} | {seeds[0]['params']:,} | "
                     f"{statistics.mean(s['min_per_epoch'] for s in seeds):.2f} |")
    text = "\n".join(lines) if runs else "no finished dev runs yet (./run dev u1 | m3)"
    print(text)
    if runs:
        (load_paths().runs / "dev_summary.md").write_text(text + "\n")


if __name__ == "__main__":
    main()
