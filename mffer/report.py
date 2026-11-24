"""Results ledger, paper tables and figures, generated only from the run directories.

    python -m mffer.report

Writes into the paper project (`paper` in configs/paths.yaml); nothing there is typed by hand:
- notes/results-ledger.md: one row per run directory, with its status;
- generated/tab-main.tex (B0, U1, U2, C1, C2), tab-ladder.tex (D1-D4, C2), tab-federated.tex (C2, F1, F2,
  L1, F3, F4), tab-sex.tex (valence macro-F1 by sex): `tabular` bodies only, so captions and labels stay
  in the section text (`\\input{generated/tab-main}`). Each lists the run IDs behind it in a comment;
- generated/fig-curves.pdf: val valence macro-F1 per epoch (or round), one line per counted run.

Columns follow the metric priority (metrics.PRIORITY, ADR 0014): valence macro-F1, displayed-intensity MAE,
perceived-intensity MAE, then the secondary scores.

Which runs count: per (config, split), the latest finished run trained on the CPU or CUDA (mffer.device)
that selected its epoch with the current monitor (metrics.MONITOR; a trained run whose config.yaml names
another monitor is "invalid (old monitor)") under the current protocol (train.PROTOCOL; a trained run of an
older one is "invalid (old protocol)", ADR 0017). Dev runs (val.json, no test) are "dev (val only)" and never
count. Earlier finished runs are "superseded"; Metal GPU runs are
"invalid (GPU)" (the Metal GPU diverges, ADR 0001). Runs from before 19 Nov have no `device`: a GPU among
them was Metal.
An experiment's score on a split is the mean over its configs (L1: its two clients, both required); the
tables give mean ± sd over splits. Gaps to C2 are paired: mean ± sd of the per-split differences.
"""
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from mffer.config import load_paths
from mffer.metrics import MONITOR, PRIORITY
from mffer.train import PROTOCOL

RUN_DIR = re.compile(r"^(?P<config>.+)-s(?P<split>\d)-(?P<stamp>\d{8}-\d{4})$")
EXPERIMENTS = {  # README ID: (description, configs averaged per split)
    "B0": ("Majority class / constant", ["b0_majority"]),
    "U1": ("Audio-only", ["u1_audio"]),
    "U2": ("Video-only", ["u2_video"]),
    "C1": ("M1, audio-visual", ["c1_m1"]),
    "C2": ("M3, full model", ["c2_m3"]),
    "D1": ("M1 + deep visual", ["d1"]),
    "D2": ("D1 + gating", ["d2"]),
    "D3": ("D2 + demographics", ["d3"]),
    "D4": ("D2 + text", ["d4"]),
    "F1": ("FL-IID, 2 clients", ["f1_iid"]),
    "F2": ("FL-non-IID, 2 clients", ["f2_actor"]),
    "F3": ("FL-non-IID, 4 clients", ["f3_actor4"]),
    "F4": ("FedProx, 2 clients", ["f4_fedprox"]),
    "L1": ("Local-only, mean of 2 clients", ["l1_local_c0", "l1_local_c1"]),
}
CONFIG_ID = {config: eid for eid, (_, configs) in EXPERIMENTS.items() for config in configs}
# name: (rows, splits used or None = all). The ladder runs on split 0 only, so its C1 and C2 rows are
# restricted to split 0 too: neighbouring rows then differ by one design change, not also by data.
TABLES = {"main": (["B0", "U1", "U2", "C1", "C2"], None), "ladder": (["C1", "D1", "D2", "D3", "D4", "C2"], [0]),
          "federated": (["C2", "F1", "F2", "L1", "F3", "F4"], None)}
# (column, LaTeX header, kind): kind "pct" = share shown in %, "mae" = 3 decimals, "num" = 1 decimal
# Priority order first (primary, then the two intensities), then secondary scores. Valence accuracy stays
# visible because it shows why it is not the primary: always answering "negative" scores 68%.
METRICS = [
    ("displayed_valence.macro_f1", "Valence F1", "pct"),
    ("displayed_intensity.mae", "Displ.\\ int.\\ MAE", "mae"),
    ("perceived_intensity.mae", "Perc.\\ int.\\ MAE", "mae"),
    ("displayed_valence.accuracy", "Valence acc", "pct"),
    ("displayed_emotion.macro_f1", "Displ.\\ emo.\\ F1", "pct"),
    ("perceived_emotion.macro_f1", "Perc.\\ emo.\\ F1", "pct"),
]
assert [key for key, _, _ in METRICS[:len(PRIORITY)]] == PRIORITY
PRIMARY = PRIORITY[0]
SEX = [("displayed_valence.macro_f1_by_sex.Female", "Female", "pct"),
       ("displayed_valence.macro_f1_by_sex.Male", "Male", "pct")]


# ---------------------------------------------------------------- loading

def load_runs(runs_dir: Path) -> pd.DataFrame:
    """One row per run directory: identity, status, flattened test metrics, cost."""
    records = []
    for path in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        match = RUN_DIR.match(path.name)
        if not match:
            continue
        meta = json.loads((path / "meta.json").read_text()) if (path / "meta.json").exists() else {}
        result = json.loads((path / "test.json").read_text()) if (path / "test.json").exists() else None
        dev = (path / "val.json").exists()
        config = yaml.safe_load((path / "config.yaml").read_text()) if (path / "config.yaml").exists() else {}
        record = {"run": path.name, "config": match["config"], "id": CONFIG_ID.get(match["config"]),
                  "split": int(match["split"]), "git": meta.get("git"), "device": meta.get("device", "metal" if meta.get("gpu") else "cpu"),
                  "parameters": meta.get("parameters"), "update_bytes": meta.get("update_bytes"),
                  "train_minutes": meta.get("train_minutes"), "finished": result is not None,
                  "old_monitor": config.get("monitor", MONITOR) != MONITOR,   # B0 trains nothing: no monitor
                  "old_protocol": isinstance(config.get("model"), dict) and config.get("protocol", 1) < PROTOCOL,
                  "dev": dev}
        if result:
            record.update(best_epoch=result.get("best_epoch"), epochs_run=result.get("epochs_run"),
                          **pd.json_normalize(result["test"]).iloc[0].to_dict())
        records.append(record)
    runs = pd.DataFrame(records)
    if runs.empty:
        return runs
    runs["status"] = "incomplete"
    runs.loc[runs.dev, "status"] = "dev (val only)"
    runs.loc[runs.finished & runs.old_protocol, "status"] = "invalid (old protocol)"
    runs.loc[runs.finished & runs.old_monitor, "status"] = "invalid (old monitor)"
    metal = ~runs.device.isin(["cpu", "cuda"])
    runs.loc[runs.finished & metal, "status"] = "invalid (GPU)"
    valid = runs.finished & ~metal & ~runs.old_monitor & ~runs.old_protocol
    latest = runs[valid].groupby(["config", "split"]).run.max()   # names sort by time stamp
    runs.loc[valid, "status"] = "superseded"
    runs.loc[runs.run.isin(latest), "status"] = "counted"
    if "epochs_run" in runs:
        runs["min_per_epoch"] = runs.train_minutes / runs.epochs_run
    return runs


def per_split(runs: pd.DataFrame, eid: str, splits: list[int] | None = None) -> pd.DataFrame:
    """Score of experiment `eid` on each split where all its configs have a counted run (mean over them)."""
    configs = EXPERIMENTS[eid][1]
    counted = runs[(runs.status == "counted") & runs.config.isin(configs)] if len(runs) else runs
    if splits is not None and len(counted):
        counted = counted[counted.split.isin(splits)]
    if counted.empty:
        return pd.DataFrame()
    complete = counted.groupby("split").config.nunique() == len(configs)
    counted = counted[counted.split.isin(complete[complete].index)]
    numeric = counted.select_dtypes("number").drop(columns=["split"], errors="ignore")
    out = numeric.groupby(counted.split).mean()
    out["runs"] = counted.groupby("split").run.apply(list)
    return out


# ---------------------------------------------------------------- formatting

def fmt(values: list[float], kind: str, latex: bool = True) -> str:
    values = [v for v in values if pd.notna(v)]
    if not values:
        return "--" if latex else "–"
    scale, digits = {"pct": (100, 1), "mae": (1, 3), "num": (1, 1)}[kind]
    mean = f"{scale * statistics.mean(values):.{digits}f}"
    if len(values) == 1:
        return mean
    sd = f"{scale * statistics.stdev(values):.{digits}f}"
    return f"{mean} $\\pm$ {sd}" if latex else f"{mean} ± {sd}"


def minutes(values: list[float]) -> str:
    """Cost column: mean only (its sd across splits is not a result)."""
    values = [v for v in values if pd.notna(v)]
    if not values:
        return "--"
    mean = statistics.mean(values)
    return "$<$0.1" if mean < 0.1 else f"{mean:.1f}"


def column(scores: pd.DataFrame, key: str) -> list[float]:
    return scores[key].tolist() if key in scores else []


def tabular(rows: list[list[str]], header: list[str], align: str, sources: list[str],
            splits: list[int] | None = None) -> str:
    used = "all actor splits" if splits is None else f"split {', '.join(map(str, splits))} only"
    lines = [f"% Generated by `python -m mffer.report` on {datetime.now():%d %b %Y %H:%M}; do not edit.",
             f"% Mean $\\pm$ sd over splits (n = splits; {used}). Runs:"]
    lines += [f"%   {run}" for run in sources] or ["%   (none yet)"]
    lines += [f"\\begin{{tabular}}{{@{{}}{align}@{{}}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(row) + " \\\\" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def results_table(runs: pd.DataFrame, ids: list[str], federated: bool = False,
                  splits: list[int] | None = None) -> str:
    """Federated: all rows are M3, so Params gives way to the update size and the paired gap to C2."""
    reference = per_split(runs, "C2", splits)
    header = ["ID", "Configuration", "n"] + [h for _, h, _ in METRICS]
    header += ["Min/ep.", "Update (MB)", "$\\Delta$ valence F1 vs C2 (pp)"] if federated else ["Params", "Min/ep."]
    rows, sources = [], []
    for eid in ids:
        scores = per_split(runs, eid, splits)
        sources += sum(scores.runs.tolist(), []) if len(scores) else []
        row = [eid, EXPERIMENTS[eid][0], str(len(scores))]
        row += [fmt(column(scores, key), kind) for key, _, kind in METRICS]
        params = column(scores, "parameters")
        if not federated:
            row.append(f"{int(params[0]):,}" if params and pd.notna(params[0]) else "--")
        row.append(minutes(column(scores, "min_per_epoch")))
        if federated:
            update = column(scores, "update_bytes")
            row.append(f"{update[0] / 1e6:.2f}" if update and pd.notna(update[0]) else "--")
            key = PRIMARY
            paired = [] if eid == "C2" or reference.empty or scores.empty else \
                (scores[key] - reference[key]).dropna().tolist()
            row.append("--" if not paired else fmt(paired, "pct"))
        rows.append(row)
    align = "llr" + "c" * len(METRICS) + ("rrc" if federated else "rr")
    return tabular(rows, header, align, sources, splits)


def sex_table(runs: pd.DataFrame) -> str:
    rows, sources = [], []
    for eid in EXPERIMENTS:
        scores = per_split(runs, eid)
        if scores.empty:
            continue
        sources += sum(scores.runs.tolist(), [])
        gap = (scores[SEX[0][0]] - scores[SEX[1][0]]).tolist() if all(k in scores for k, _, _ in SEX) else []
        rows.append([eid, EXPERIMENTS[eid][0], str(len(scores))] +
                    [fmt(column(scores, key), kind) for key, _, kind in SEX] + [fmt(gap, "pct")])
    return tabular(rows, ["ID", "Configuration", "n", "Female", "Male", "F $-$ M (pp)"], "llrccc", sources)


def ledger(runs: pd.DataFrame) -> str:
    keys = [("Val. F1", PRIMARY, "pct"), ("Displ. int. MAE", "displayed_intensity.mae", "mae"),
            ("Perc. int. MAE", "perceived_intensity.mae", "mae"), ("Val. acc", "displayed_valence.accuracy", "pct"),
            ("Displ. emo. F1", "displayed_emotion.macro_f1", "pct"), ("Perc. emo. F1", "perceived_emotion.macro_f1", "pct"),
            ("Val. F1 female", SEX[0][0], "pct"), ("Val. F1 male", SEX[1][0], "pct")]
    lines = ["# Results ledger", "",
             f"Generated by `python -m mffer.report` (mffer-exp) on {datetime.now():%d %b %Y %H:%M}; do not edit.",
             "One row per run directory. Only **counted** runs enter the paper tables: the latest finished",
             "CPU or CUDA run per configuration and split, selected on val valence macro-F1. Test scores at the",
             "val-selected epoch (or round).", "",
             "| Run ID | Exp. | Split | Status | Git | Device | Best / epochs | Min | " +
             " | ".join(k for k, _, _ in keys) + " |",
             "|" + "---|" * (8 + len(keys))]
    for _, r in runs.sort_values(["id", "config", "split", "run"], na_position="last").iterrows():
        cells = [f"`{r['run']}`", r["id"] if isinstance(r["id"], str) else "?", str(r["split"]), r["status"], f"`{r['git']}`", r["device"],
                 f"{r.get('best_epoch'):.0f} / {r.get('epochs_run'):.0f}" if r["finished"] else "–",
                 f"{r['train_minutes']:.0f}" if pd.notna(r["train_minutes"]) else "–"]
        cells += [fmt([r[key]], kind, latex=False) if key in r else "–" for _, key, kind in keys]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def curves_figure(runs: pd.DataFrame, runs_dir: Path, out: Path) -> None:
    """Two panels, greyscale (the paper is printed in black and white): centralised models and the
    federated comparison. One thin line per counted run (split), line style per experiment."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    panels = [("Centralised", ["U1", "C1", "C2"]), ("Centralised vs federated", ["C2", "F1", "F2", "L1"])]
    styles = ["-", "--", ":", "-."]
    greys = ["0.0", "0.35", "0.55", "0.7"]
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 1.9), sharey=True, constrained_layout=True)
    for ax, (title, ids) in zip(axes, panels, strict=True):
        for i, eid in enumerate(ids):
            counted = runs[(runs.status == "counted") & runs.config.isin(EXPERIMENTS[eid][1])] if len(runs) else runs
            for j, run in enumerate(counted.run):
                curve = pd.read_csv(runs_dir / run / "epochs.csv")[MONITOR]
                ax.plot(range(1, len(curve) + 1), 100 * curve, styles[i], color=greys[i], linewidth=0.8,
                        label=eid if j == 0 else None)
        ax.set_title(title)
        ax.set_xlabel("Epoch (FL: round)")
        ax.grid(color="0.9", linewidth=0.5)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False)
    axes[0].set_ylabel("Val valence macro-F1 (%)")
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    paths = load_paths()
    runs = load_runs(paths.runs)
    out = paths.paper / "generated"
    out.mkdir(exist_ok=True)
    (paths.paper / "notes" / "results-ledger.md").write_text(ledger(runs))
    for name, (ids, splits) in TABLES.items():
        (out / f"tab-{name}.tex").write_text(results_table(runs, ids, federated=name == "federated", splits=splits))
    (out / "tab-sex.tex").write_text(sex_table(runs))
    curves_figure(runs, paths.runs, out / "fig-curves.pdf")
    counted = (runs.status == "counted").sum() if len(runs) else 0
    print(f"{len(runs)} run directories, {counted} counted -> {paths.paper / 'notes/results-ledger.md'}, {out}/")


if __name__ == "__main__":
    main()
