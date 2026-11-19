"""Render the scores of every run in runs/ into results.html (self-contained; open it in a browser).

The run directories are the only source. `./run` calls this after every run; by hand (mffer env):
`conda run -n mffer python scripts/render_results.py`. Which runs count comes from `mffer.report.load_runs`,
so this page and the paper tables always agree (latest finished CPU run per config and split, selected on
the current monitor).

The page: a summary per experiment (mean ± sd over splits), a list of runs, and for the selected run its
test scores, learning curves (hover, drag to zoom, expand, log scale), confusion matrices, per-class and
per-sex scores, and every epoch as a table.
"""
import csv
import html
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

from mffer.report import load_runs as report_runs

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = re.compile(r"^(?P<config>.+)-s(?P<split>\d)-(?P<stamp>\d{8}-\d{4})$")
EXPERIMENTS = {  # config name -> (README ID, description), in README order (B0 has its own row)
    "u1_audio": ("U1", "Audio-only"), "u2_video": ("U2", "Video-only"),
    "c1_m1": ("C1", "M1 audio-visual baseline"), "c2_m3": ("C2", "M3 full model"),
    "d1": ("D1", "M1 + deeper visual encoder"), "d2": ("D2", "D1 + gating"),
    "d3": ("D3", "D2 + demographics"), "d4": ("D4", "D2 + text"),
    "f1_iid": ("F1", "FL-IID, 2 clients"), "f2_actor": ("F2", "FL-non-IID, 2 clients"),
    "f3_actor4": ("F3", "FL-non-IID, 4 clients"), "f4_fedprox": ("F4", "FedProx, 2 clients"),
    "l1_local_c0": ("L1", "Local-only, client 0"), "l1_local_c1": ("L1", "Local-only, client 1"),
}
B0_CONFIG = "b0_majority"   # python -m mffer.baselines; its mean over splits is the "≤ B0" reference
BASELINE: dict[str, float] = {}   # "<head>.<metric>" -> B0 mean over splits
CLASSES = {  # fixed order, as in mffer/data/cremad.py
    "displayed_emotion": ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"],
    "perceived_emotion": ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"],
    "displayed_valence": ["negative", "neutral", "positive"],
}
# (label, head, metric, higher is better), in the paper's priority order (metrics.PRIORITY, ADR 0014)
COLUMNS = [
    ("Valence F1 (primary)", "displayed_valence", "macro_f1", True),
    ("Displayed intensity MAE", "displayed_intensity", "mae", False),
    ("Perceived intensity MAE", "perceived_intensity", "mae", False),
    ("Valence acc", "displayed_valence", "accuracy", True),
    ("Displayed emotion F1", "displayed_emotion", "macro_f1", True),
    ("Displayed emotion acc", "displayed_emotion", "accuracy", True),
    ("Perceived emotion F1", "perceived_emotion", "macro_f1", True),
    ("Perceived emotion acc", "perceived_emotion", "accuracy", True),
]
RUN_COLUMNS = COLUMNS[:4]   # runs list: the prioritised scores only


def confusion(path: Path) -> dict:
    """Per classification head: confusion matrix (rows = true, columns = predicted) and per-class scores."""
    if not path.exists():
        return {}
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for head, labels in CLASSES.items():
        if f"{head}_mask" not in (rows[0] if rows else {}):
            continue
        scored = [r for r in rows if r[f"{head}_mask"] == "1"]
        labels = labels + sorted({r[f"{head}_{k}"] for r in scored for k in ("true", "pred")} - set(labels))
        index = {c: i for i, c in enumerate(labels)}
        matrix = [[0] * len(labels) for _ in labels]
        for r in scored:
            matrix[index[r[f"{head}_true"]]][index[r[f"{head}_pred"]]] += 1
        per_class = []
        for i, label in enumerate(labels):
            tp, support, predicted = matrix[i][i], sum(matrix[i]), sum(row[i] for row in matrix)
            precision = tp / predicted if predicted else 0.0
            recall = tp / support if support else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_class.append({"label": label, "precision": precision, "recall": recall, "f1": f1,
                              "support": support, "predicted": predicted})
        out[head] = {"labels": labels, "matrix": matrix, "per_class": per_class}
    return out


def load_runs() -> list[dict]:
    runs = []
    for path in sorted((ROOT / "runs").iterdir()):
        match = RUN_DIR.match(path.name)
        if not (path.is_dir() and match):
            continue
        eid, desc = EXPERIMENTS.get(match["config"], (match["config"], ""))
        run = {"dir": path.name, "config": match["config"], "id": eid, "desc": desc,
               "split": int(match["split"]), "stamp": match["stamp"], "test": None, "best_epoch": None,
               "epochs": {}, "meta": {}, "confusion": {}}
        if (path / "epochs.csv").exists():
            with open(path / "epochs.csv") as f:
                table = list(csv.DictReader(f))
            run["epochs"] = {k: [float(r[k]) if r.get(k) else None for r in table]
                             for k in (table[0] if table else {}) if k != "epoch"}
        if (path / "meta.json").exists():
            meta = json.loads((path / "meta.json").read_text())
            run["meta"] = {k: meta.get(k) for k in ("seed", "git", "parameters", "train_minutes", "started",
                                                     "finished", "machine", "tensorflow", "keras")}
        if (path / "test.json").exists():
            result = json.loads((path / "test.json").read_text())
            run["test"], run["best_epoch"] = result["test"], result.get("best_epoch")
            run["confusion"] = confusion(path / "predictions.csv")
        runs.append(run)
    table = report_runs(ROOT / "runs")
    status = dict(zip(table.run, table.status, strict=True)) if len(table) else {}
    for run in runs:
        run["status"] = status.get(run["dir"], "incomplete")
        run["state"] = run["status"].split()[0]   # CSS class: counted, superseded, incomplete, invalid
        run["warning"] = loss_warning(run)
    return runs


def metric(run: dict, head: str, name: str) -> float | None:
    return (run["test"] or {}).get(head, {}).get(name)


def fmt(value: float | None, mae: bool) -> str:
    if value is None:
        return "–"
    return f"{value:.3f}" if mae else f"{100 * value:.1f}%"


def loss_warning(run: dict) -> str:
    losses = [v for v in run["epochs"].get("loss", []) if v is not None]
    if len(losses) < 2 or losses[-1] < 1.5 * min(losses):
        return ""
    return f"train loss rose {min(losses):.1f} → {losses[-1]:.1f}"


def summary_table(runs: list[dict]) -> str:
    head = "".join(f"<th>{html.escape(c[0])}</th>" for c in COLUMNS)
    base = "".join(f"<td>{fmt(BASELINE.get(f'{h}.{m}'), m == 'mae')}</td>" for _, h, m, _ in COLUMNS)
    n_b0 = sum(r["config"] == B0_CONFIG and r["status"] == "counted" for r in runs)
    rows = [f'<tr class="baseline"><td>B0</td><td>Majority class / constant <span class="sd">(mean over splits)'
            f"</span></td><td>{n_b0} / 3</td>{base}</tr>"]
    for config, (eid, desc) in EXPERIMENTS.items():
        counted = [r for r in runs if r["config"] == config and r["status"] == "counted"]
        if not counted:
            rows.append(f'<tr class="empty"><td>{eid}</td><td>{desc}</td><td>0 / 3</td>'
                        f'<td colspan="{len(COLUMNS)}">not run yet</td></tr>')
            continue
        cells = []
        for _, h, m, higher in COLUMNS:
            values = [v for v in (metric(r, h, m) for r in counted) if v is not None]
            if not values:
                cells.append("<td>–</td>")
                continue
            mean = statistics.mean(values)
            sd = f' <span class="sd">± {fmt(statistics.stdev(values), m == "mae")}</span>' if len(values) > 1 else ""
            flag = ""
            b0 = BASELINE.get(f"{h}.{m}")
            if b0 is not None and not (mean > b0 if higher else mean < b0):
                flag = ' <span class="flag" title="not better than the B0 baseline">≤ B0</span>'
            cells.append(f"<td>{fmt(mean, m == 'mae')}{sd}{flag}</td>")
        splits = ", ".join(str(r["split"]) for r in sorted(counted, key=lambda r: r["split"]))
        rows.append(f"<tr><td>{eid}</td><td>{desc}</td><td>{len(counted)} / 3 <span class='sd'>"
                    f"(split {splits})</span></td>{''.join(cells)}</tr>")
    return (f"<table><thead><tr><th>ID</th><th>Experiment</th><th>Splits</th>{head}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def runs_table(runs: list[dict]) -> str:
    head = "".join(f"<th>{html.escape(c[0])}</th>" for c in RUN_COLUMNS)
    rows = []
    for run in sorted(runs, key=lambda r: r["dir"], reverse=True):
        epochs = len(run["epochs"].get("loss", []))
        minutes = run["meta"].get("train_minutes")
        cells = "".join(f"<td>{fmt(metric(run, h, m), m == 'mae')}</td>" for _, h, m, _ in RUN_COLUMNS)
        warning = f'<span class="flag">⚠ {html.escape(run["warning"])}</span>' if run["warning"] else ""
        rows.append(
            f'<tr class="run {run["state"]}" data-run="{html.escape(run["dir"])}" tabindex="0">'
            f'<td>{run["id"]}</td><td>{run["split"]}</td>'
            f'<td><code>{html.escape(run["dir"])}</code></td>'
            f'<td><span class="badge {run["state"]}">{run["status"]}</span> {warning}</td>'
            f'<td>{run["best_epoch"] or "–"} / {epochs}</td>'
            f'<td>{f"{minutes:.0f}" if minutes is not None else "–"}</td>{cells}</tr>')
    return (f"<table class='runs'><thead><tr><th>ID</th><th>Split</th><th>Run</th><th>Status</th>"
            f"<th>Best / epochs</th><th>Min</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Training scores</title>
<style>
:root { --surface: #fcfcfb; --raised: #f3f2ef; --line: #e2e1dc; --grid: #ecebe7; --text: #0b0b0b;
  --text-2: #52514e; --text-3: #6f6e69; --train: #2a78d6; --val: #eb6834; --seq: #2a78d6; --flag: #b3261e;
  --good: #0a7a0a; --select: rgba(42, 120, 214, .12); color-scheme: light; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --surface: #1a1a19; --raised: #242423;
  --line: #383835; --grid: #2c2c2a; --text: #f0efec; --text-2: #c3c2b7; --text-3: #9d9c94; --train: #3987e5;
  --val: #d95926; --seq: #3987e5; --flag: #f28b82; --good: #5fcf5f; --select: rgba(57, 135, 229, .18);
  color-scheme: dark; } }
:root[data-theme="dark"] { --surface: #1a1a19; --raised: #242423; --line: #383835; --grid: #2c2c2a;
  --text: #f0efec; --text-2: #c3c2b7; --text-3: #9d9c94; --train: #3987e5; --val: #d95926; --seq: #3987e5;
  --flag: #f28b82; --good: #5fcf5f; --select: rgba(57, 135, 229, .18); color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 16px 64px; background: var(--surface); color: var(--text);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
main { max-width: 1400px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 17px; margin: 36px 0 6px; }
h3 { font-size: 15px; margin: 28px 0 6px; }
p.note { color: var(--text-2); margin: 0 0 10px; max-width: 900px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 10px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap;
  vertical-align: middle; }
th { font-weight: 600; color: var(--text-2); font-size: 12px; background: var(--surface); white-space: normal; }
th:nth-child(-n+3), td:nth-child(-n+3) { text-align: left; }
tr.baseline td { background: var(--raised); }
tr.empty td { color: var(--text-3); } tr.empty td[colspan] { text-align: left; }
table.runs tr.run { cursor: pointer; }
table.runs tr.run:hover td { background: var(--raised); }
table.runs tr.run.selected td { background: var(--select); }
table.runs tr.run:focus-visible { outline: 2px solid var(--train); outline-offset: -2px; }
table.runs tr.superseded td, table.runs tr.incomplete td, table.runs tr.invalid td { color: var(--text-3); }
table.runs td:nth-child(4) { text-align: left; }
.sd { color: var(--text-2); font-size: 12px; }
.flag { color: var(--flag); font-size: 12px; font-weight: 600; }
.badge { font-size: 11px; font-weight: 600; padding: 1px 7px; border-radius: 9px; border: 1px solid var(--line);
  color: var(--text-2); }
.badge.counted { color: var(--good); border-color: currentColor; }
code { font-size: 12px; }
button, select, input { font: inherit; font-size: 13px; color: var(--text); background: var(--surface);
  border: 1px solid var(--line); border-radius: 6px; padding: 4px 10px; }
button { cursor: pointer; } button:hover { background: var(--raised); }
button[aria-pressed="true"] { background: var(--select); border-color: var(--train); }
input[type=number] { width: 64px; padding: 4px 6px; }
/* run detail */
#detail { margin-top: 8px; border-top: 2px solid var(--line); }
#detail .head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 16px; margin: 20px 0 4px; }
#detail .head h2 { margin: 0; }
.facts { display: flex; flex-wrap: wrap; gap: 4px 20px; color: var(--text-2); font-size: 13px; margin: 6px 0 0; }
.facts b { color: var(--text); font-weight: 600; }
.tiles { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; margin-top: 14px; }
.tile { background: var(--raised); border-radius: 10px; padding: 12px 14px; }
.tile .label { font-size: 12px; color: var(--text-2); }
.tile .value { font-size: 26px; font-weight: 650; font-variant-numeric: tabular-nums; line-height: 1.2; }
.tile .sub { font-size: 12px; color: var(--text-2); font-variant-numeric: tabular-nums; }
.tile .delta.up { color: var(--good); } .tile .delta.down { color: var(--flag); }
.toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; margin: 10px 0 12px;
  padding: 8px 12px; background: var(--raised); border-radius: 8px; font-size: 13px; }
.toolbar .hint { color: var(--text-2); }
.legend { display: inline-flex; gap: 14px; align-items: center; }
.key { display: inline-flex; align-items: center; gap: 6px; }
.key i { width: 18px; height: 3px; border-radius: 2px; display: inline-block; }
.key i.best { height: 12px; width: 0; border-left: 2px dashed var(--text-3); border-radius: 0; }
.charts { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 14px; grid-auto-flow: dense; }
.chart { position: relative; border: 1px solid var(--line); border-radius: 10px; padding: 10px 10px 4px; }
.chart.wide { grid-column: 1 / -1; }
.chart .ctitle { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.chart .ctitle b { font-size: 13px; }
.chart .ctitle .what { font-size: 12px; color: var(--text-2); font-weight: 400; }
.chart .ctitle button { padding: 1px 8px; font-size: 12px; }
.chart svg { display: block; width: 100%; user-select: none; touch-action: none; }
.chart svg text { fill: var(--text-2); font-size: 11px; font-variant-numeric: tabular-nums; }
.chart svg .grid { stroke: var(--grid); }
.chart svg .axis { stroke: var(--line); }
.chart svg .line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.chart svg .dot { stroke: var(--surface); stroke-width: 2; }
.chart svg .best { stroke: var(--text-3); stroke-width: 1.5; stroke-dasharray: 4 3; }
.chart svg .cross { stroke: var(--text-3); stroke-width: 1; }
.chart svg .brush { fill: var(--select); stroke: var(--train); stroke-width: 1; }
.chart svg text.dl { font-size: 12px; font-weight: 600; fill: var(--text); }
.chart svg text.bestlabel { fill: var(--text-3); }
.tip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--line);
  border-radius: 8px; padding: 6px 10px; font-size: 12px; box-shadow: 0 4px 14px rgba(0,0,0,.12);
  white-space: nowrap; font-variant-numeric: tabular-nums; z-index: 2; }
.tip .row { display: flex; align-items: center; gap: 6px; }
.tip .row i { width: 10px; height: 3px; border-radius: 2px; }
.tip .row span:last-child { margin-left: auto; padding-left: 12px; font-weight: 600; }
details { margin-top: 18px; } summary { cursor: pointer; font-weight: 600; font-size: 15px; }
.grid2 { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 18px 28px; }
table.cm td, table.cm th { text-align: center; padding: 5px 7px; font-size: 12px; }
table.cm td.cell { min-width: 52px; }
table.cm td.diag { outline: 2px solid var(--text); outline-offset: -2px; }
table.cm th.rowh { text-align: right; }
table.small td, table.small th { font-size: 12px; padding: 4px 8px; }
table.small th:first-child, table.small td:first-child { text-align: left; }
table.epochs td.bestrow { font-weight: 650; background: var(--select); }
.empty-note { color: var(--text-2); padding: 18px 0; }
@media (max-width: 520px) { .charts, .grid2 { grid-template-columns: 1fr; } }
</style></head>
<body><main>
<h1>Training scores</h1>
<p class="note">From <code>runs/</code>, updated __UPDATED__. Every test score is measured once, at the epoch with the
best validation valence macro-F1 (the primary metric). <b>Train</b> = the actors the model learns from; <b>val</b> = held-out
actors used only to pick that epoch; <b>test</b> = other held-out actors, scored at the end.</p>

<h2>By experiment</h2>
<p class="note">Mean ± sd over the latest finished run of each split. <span class="flag">≤ B0</span> marks a score no
better than predicting the majority class / a constant.</p>
<div class="scroll">__SUMMARY__</div>

<h2>Runs</h2>
<p class="note">Newest first. <b>Click a run</b> (or Tab to it and press Enter) for its learning curves and full test
breakdown. Superseded, incomplete and invalid runs (GPU, or epoch picked by an old monitor) are greyed out
and not counted above.</p>
<div class="scroll">__RUNS__</div>

<section id="detail" aria-live="polite"><p class="empty-note">Select a run above.</p></section>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("data").textContent);
const RUNS = Object.fromEntries(DATA.runs.map(r => [r.dir, r]));
const HEADS = {
  displayed_emotion: "Displayed emotion", perceived_emotion: "Perceived emotion",
  displayed_valence: "Displayed valence", displayed_intensity: "Displayed intensity",
  perceived_intensity: "Perceived intensity",
};
const pct = v => v == null ? "–" : (100 * v).toFixed(1) + "%";
const num = (v, d = 3) => v == null ? "–" : v.toFixed(d);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));

// Learning-curve charts: one measure each, train vs val on one axis.
const CHARTS = [
  {key: "loss", title: "Total loss", what: "lower is better", fmt: v => v == null ? "–" : v >= 10 ? v.toFixed(1) : v.toFixed(2), loss: true, wide: true},
  {key: "displayed_valence_macro_f1", title: "Valence macro-F1 (primary)", what: "higher is better; picks the best epoch", fmt: pct},
  {key: "displayed_intensity_mae", title: "Displayed intensity MAE", what: "lower is better (IEO clips only)", fmt: v => num(v)},
  {key: "perceived_intensity_mae", title: "Perceived intensity MAE", what: "lower is better", fmt: v => num(v)},
  {key: "displayed_valence_accuracy", title: "Valence accuracy", what: "higher is better", fmt: pct},
  {key: "displayed_emotion_macro_f1", title: "Displayed emotion macro-F1", what: "higher is better", fmt: pct},
  {key: "perceived_emotion_macro_f1", title: "Perceived emotion macro-F1", what: "higher is better", fmt: pct},
];
const LOSS_CHARTS = Object.keys(HEADS).map(h => ({key: h + "_loss", title: HEADS[h] + " loss",
  what: h.endsWith("intensity") ? "MSE" : "cross-entropy", fmt: v => num(v, 3), loss: true}));

const state = {run: null, from: 1, to: 1, log: false, hover: null, wide: new Set()};

function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || Math.abs(hi) || 1, raw = span / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) || 10 * mag;
  const ticks = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) ticks.push(+t.toPrecision(12));
  return ticks;
}

function drawChart(box, spec) {
  const run = state.run, ep = run.epochs, n = (ep.loss || []).length;
  const series = [{name: "Train", cls: "train", vals: ep[spec.key] || []},
                  {name: "Val", cls: "val", vals: ep["val_" + spec.key] || []}];
  const W = box.clientWidth - 20, H = box.classList.contains("wide") ? 300 : 210;
  const m = {l: 52, r: 54, t: 26, b: 30}, iw = W - m.l - m.r, ih = H - m.t - m.b;
  const from = state.from, to = Math.max(state.to, from + 1e-9);
  const log = spec.loss && state.log;
  const visible = series.flatMap(s => s.vals.filter((v, i) => v != null && i + 1 >= from && i + 1 <= to && (!log || v > 0)));
  const svg = box.querySelector("svg");
  if (!visible.length) { svg.setAttribute("viewBox", `0 0 ${W} 60`); svg.innerHTML = `<text x="8" y="30">no values</text>`; return; }
  let lo = Math.min(...visible), hi = Math.max(...visible);
  const tf = log ? Math.log10 : v => v, inv = log ? v => 10 ** v : v => v;
  let a = tf(lo), b = tf(hi); const pad = (b - a || Math.abs(b) || 1) * 0.08; a -= pad; b += pad;
  if (!log && lo >= 0 && a < 0) a = 0;
  const x = e => m.l + (to === from ? iw / 2 : (e - from) / (to - from) * iw);
  const y = v => m.t + ih - (tf(v) - a) / (b - a) * ih;
  let g = "";
  // y grid + labels
  let yt = niceTicks(inv(a), inv(b));
  if (log) {   // 1-2-5 steps across decades; a narrow zoomed range keeps plain nice ticks
    const decades = [];
    for (let k = Math.floor(a); k <= Math.ceil(b); k++) for (const f of [1, 2, 5]) decades.push(f * 10 ** k);
    const inside = decades.filter(v => Math.log10(v) >= a && Math.log10(v) <= b);
    yt = inside.length >= 3 ? inside : yt.filter(v => v > 0);
  }
  for (const t of yt) { const yy = y(t); if (yy < m.t - 1 || yy > m.t + ih + 1) continue;
    g += `<line class="grid" x1="${m.l}" x2="${m.l + iw}" y1="${yy}" y2="${yy}"/><text x="${m.l - 6}" y="${yy + 4}" text-anchor="end">${spec.fmt(t)}</text>`; }
  // x ticks: integer epochs
  const span = to - from, step = Math.max(1, Math.ceil(span / Math.max(2, Math.floor(iw / 46))));
  for (let e = Math.ceil(from); e <= to; e += step)
    g += `<line class="axis" x1="${x(e)}" x2="${x(e)}" y1="${m.t + ih}" y2="${m.t + ih + 4}"/><text x="${x(e)}" y="${m.t + ih + 16}" text-anchor="middle">${e}</text>`;
  g += `<line class="axis" x1="${m.l}" x2="${m.l + iw}" y1="${m.t + ih}" y2="${m.t + ih}"/>`;
  g += `<text x="${m.l + iw / 2}" y="${H - 2}" text-anchor="middle">epoch</text>`;
  // best epoch
  if (run.best_epoch && run.best_epoch >= from && run.best_epoch <= to) {
    const bx = x(run.best_epoch);
    g += `<line class="best" x1="${bx}" x2="${bx}" y1="${m.t}" y2="${m.t + ih}"/><text class="bestlabel" x="${bx}" y="${m.t - 8}" text-anchor="middle">best epoch ${run.best_epoch} · test scored here</text>`;
  }
  g += `<clipPath id="c-${spec.key}"><rect x="${m.l - 6}" y="${m.t - 6}" width="${iw + 12}" height="${ih + 12}"/></clipPath><g clip-path="url(#c-${spec.key})">`;
  const ends = [];
  for (const s of series) {
    const pts = s.vals.map((v, i) => [i + 1, v]).filter(([e, v]) => v != null && (!log || v > 0));
    if (!pts.length) continue;
    // include one point beyond each edge so lines run to the frame when zoomed
    const inView = pts.filter(([e]) => e >= from - 1 && e <= to + 1);
    g += `<polyline class="line" stroke="var(--${s.cls})" points="${inView.map(([e, v]) => `${x(e)},${y(v)}`).join(" ")}"/>`;
    for (const [e, v] of inView) if (e >= from && e <= to)
      g += `<circle class="dot" cx="${x(e)}" cy="${y(v)}" r="${state.hover === e ? 5.5 : 4}" fill="var(--${s.cls})"/>`;
    const last = pts.filter(([e]) => e <= to).pop();
    if (last && last[0] >= from) ends.push({s, yy: y(last[1]), xx: x(last[0])});
  }
  g += `</g>`;
  // direct labels at line ends, nudged apart
  ends.sort((p, q) => p.yy - q.yy);
  if (ends.length === 2 && ends[1].yy - ends[0].yy < 14) { const mid = (ends[0].yy + ends[1].yy) / 2; ends[0].yy = mid - 7; ends[1].yy = mid + 7; }
  for (const e of ends) g += `<text class="dl" x="${e.xx + 8}" y="${e.yy + 4}">${e.s.name}</text>`;
  // crosshair
  if (state.hover && state.hover >= from && state.hover <= to)
    g += `<line class="cross" x1="${x(state.hover)}" x2="${x(state.hover)}" y1="${m.t}" y2="${m.t + ih}"/>`;
  g += `<rect class="hitarea" x="${m.l}" y="${m.t}" width="${iw}" height="${ih}" fill="transparent"/>`;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("height", H); svg.innerHTML = g;
  svg.setAttribute("aria-label", `${spec.title}: train ${spec.fmt(series[0].vals.at(-1))}, val ${spec.fmt(series[1].vals.at(-1))} at epoch ${n}`);
  box._geom = {x, m, iw, ih, from, to, n, series, spec};
}

function epochAt(box, clientX) {
  const gm = box._geom, r = box.querySelector("svg").getBoundingClientRect();
  const W = +box.querySelector("svg").getAttribute("viewBox").split(" ")[2];
  const px = (clientX - r.left) * W / r.width;
  const e = gm.from + (px - gm.m.l) / gm.iw * (gm.to - gm.from);
  return Math.min(gm.n, Math.max(1, Math.round(e)));
}

function showTip(box, e, clientX, clientY) {
  let tip = box.querySelector(".tip");
  const gm = box._geom;
  if (e == null) { if (tip) tip.remove(); return; }
  if (!tip) { tip = document.createElement("div"); tip.className = "tip"; box.appendChild(tip); }
  const best = state.run.best_epoch === e ? " · best" : "";
  tip.innerHTML = `<div><b>Epoch ${e}</b>${best}</div>` + gm.series.map(s =>
    `<div class="row"><i style="background:var(--${s.cls})"></i><span>${s.name}</span><span>${gm.spec.fmt(s.vals[e - 1])}</span></div>`).join("");
  const br = box.getBoundingClientRect();
  let left = clientX - br.left + 14; if (left + 170 > br.width) left = clientX - br.left - 180;
  tip.style.left = left + "px"; tip.style.top = Math.max(4, clientY - br.top - 40) + "px";
}

function redrawCharts() { document.querySelectorAll(".chart").forEach(box => drawChart(box, box._spec)); syncZoomInputs(); }

function attachChart(box) {
  const svg = box.querySelector("svg");
  let drag = null;
  svg.addEventListener("pointerdown", ev => { drag = {start: epochAt(box, ev.clientX), x0: ev.clientX}; svg.setPointerCapture(ev.pointerId); });
  svg.addEventListener("pointermove", ev => {
    const e = epochAt(box, ev.clientX);
    if (drag) {
      const gm = box._geom, lo = Math.min(drag.start, e), hi = Math.max(drag.start, e);
      let rect = svg.querySelector(".brush");
      if (!rect) { rect = document.createElementNS("http://www.w3.org/2000/svg", "rect"); rect.setAttribute("class", "brush"); svg.appendChild(rect); }
      rect.setAttribute("x", gm.x(lo) - 4); rect.setAttribute("width", gm.x(hi) - gm.x(lo) + 8);
      rect.setAttribute("y", gm.m.t); rect.setAttribute("height", gm.ih);
    }
    if (state.hover !== e) { state.hover = e; redrawCharts(); }
    showTip(box, e, ev.clientX, ev.clientY);
  });
  svg.addEventListener("pointerup", ev => {
    if (!drag) return;
    const e = epochAt(box, ev.clientX), lo = Math.min(drag.start, e), hi = Math.max(drag.start, e);
    const moved = Math.abs(ev.clientX - drag.x0) > 6; drag = null;
    if (moved && hi > lo) { state.from = lo; state.to = hi; }
    redrawCharts();
  });
  svg.addEventListener("pointerleave", () => { if (drag) return; state.hover = null; showTip(box, null); redrawCharts(); });
  svg.addEventListener("dblclick", resetZoom);
}

function resetZoom() { const n = (state.run.epochs.loss || []).length; state.from = 1; state.to = Math.max(n, 1); redrawCharts(); }
function syncZoomInputs() { const f = document.getElementById("zfrom"), t = document.getElementById("zto"); if (f) { f.value = state.from; t.value = state.to; } }

function chartBox(spec) {
  const box = document.createElement("figure");
  box.className = "chart" + ((spec.wide || state.wide.has(spec.key)) ? " wide" : "");
  box.style.margin = "0";
  box.innerHTML = `<figcaption class="ctitle"><span><b>${spec.title}</b> <span class="what">${spec.what}</span></span>
    <button type="button" aria-pressed="${box.classList.contains("wide")}" title="Expand / shrink this chart">${box.classList.contains("wide") ? "Shrink" : "Expand"}</button></figcaption>
    <svg role="img"></svg>`;
  box._spec = spec;
  box.querySelector("button").addEventListener("click", ev => {
    box.classList.toggle("wide"); const w = box.classList.contains("wide");
    w ? state.wide.add(spec.key) : state.wide.delete(spec.key);
    ev.target.textContent = w ? "Shrink" : "Expand"; ev.target.setAttribute("aria-pressed", w); drawChart(box, spec);
  });
  attachChart(box);
  return box;
}

function tile(label, value, sub, delta) {
  return `<div class="tile"><div class="label">${label}</div><div class="value">${value}</div>` +
    (delta ? `<div class="sub delta ${delta.cls}">${delta.text}</div>` : "") + `<div class="sub">${sub}</div></div>`;
}

function deltaVsBaseline(key, value, mae) {
  const b = DATA.baseline[key]; if (b == null || value == null) return null;
  if (mae) { const d = b - value; return {cls: d > 0 ? "up" : "down", text: `${d > 0 ? "better" : "worse"} than constant (${num(b)}) by ${num(Math.abs(d))}`}; }
  const d = 100 * (value - b);
  return {cls: d > 0 ? "up" : "down", text: `${d > 0 ? "+" : "−"}${Math.abs(d).toFixed(1)} pts vs majority (${pct(b)})`};
}

function confusionTable(head, cm) {
  const max = Math.max(...cm.matrix.map(r => Math.max(...r.map((v, j) => v / (r.reduce((a, b) => a + b, 0) || 1)))));
  let t = `<table class="cm"><caption style="text-align:left;font-weight:600;padding-bottom:4px">${HEADS[head]} <span class="sd">— rows: true class, columns: predicted; % of the row</span></caption>`;
  t += `<thead><tr><th></th>${cm.labels.map(l => `<th>${esc(l)}</th>`).join("")}<th>n</th></tr></thead><tbody>`;
  cm.matrix.forEach((row, i) => {
    const total = row.reduce((a, b) => a + b, 0);
    t += `<tr><th class="rowh">${esc(cm.labels[i])}</th>` + row.map((v, j) => {
      const share = total ? v / total : 0, mix = Math.round(share * 85);
      const ink = share > 0.5 ? "#fff" : "var(--text)";
      return `<td class="cell${i === j ? " diag" : ""}" style="background:color-mix(in oklab, var(--seq) ${mix}%, var(--surface));color:${ink}" title="true ${esc(cm.labels[i])} → predicted ${esc(cm.labels[j])}: ${v} of ${total} clips">${total ? Math.round(100 * share) + "%" : "–"}</td>`;
    }).join("") + `<td class="sd">${total}</td></tr>`;
  });
  return t + `</tbody></table>`;
}

function perClassTable(cm) {
  return `<table class="small"><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>True clips</th><th>Predicted</th></tr></thead><tbody>` +
    cm.per_class.map(c => `<tr><td>${esc(c.label)}</td><td>${pct(c.precision)}</td><td>${pct(c.recall)}</td><td>${pct(c.f1)}</td><td>${c.support}</td><td>${c.predicted}</td></tr>`).join("") +
    `</tbody></table>`;
}

function render(dir) {
  const run = RUNS[dir], el = document.getElementById("detail");
  document.querySelectorAll("tr.run").forEach(tr => tr.classList.toggle("selected", tr.dataset.run === dir));
  if (!run) { el.innerHTML = `<p class="empty-note">Select a run above.</p>`; return; }
  state.run = run; state.hover = null;
  const n = (run.epochs.loss || []).length; state.from = 1; state.to = Math.max(n, 1);
  const t = run.test || {}, meta = run.meta || {};
  const stopped = run.best_epoch && n ? (n < 90 ? `stopped early after ${n - run.best_epoch} epochs without improvement` : "ran the full 90 epochs") : "";
  let h = `<div class="head"><h2>${run.id} · split ${run.split}</h2><span class="badge ${run.state}">${run.status}</span>
    <code>runs/${esc(run.dir)}</code>${run.warning ? `<span class="flag">⚠ ${esc(run.warning)}</span>` : ""}</div>
    <div class="facts"><span>${esc(run.desc)}</span>
    <span>Best epoch <b>${run.best_epoch ?? "–"}</b> of <b>${n}</b>${stopped ? " — " + stopped : ""}</span>
    <span>Training <b>${meta.train_minutes != null ? meta.train_minutes.toFixed(1) + " min" : "–"}</b></span>
    <span>Parameters <b>${meta.parameters != null ? meta.parameters.toLocaleString() : "–"}</b></span>
    <span>Seed <b>${meta.seed ?? "–"}</b></span><span>Code <b>${esc(meta.git ?? "–")}</b></span>
    <span>Started <b>${esc((meta.started || "–").replace("T", " ").slice(0, 16))}</b></span>
    ${t.n_clips ? `<span>Test <b>${t.n_clips}</b> clips from <b>${t.n_actors}</b> actors</span>` : ""}</div>`;
  if (run.test) {
    h += `<div class="tiles">`;
    // Priority order (ADR 0014): valence macro-F1, the two intensity MAEs, then the emotion heads.
    const classTile = head => {
      const s = t[head].macro_f1_by_sex || {};
      return tile(`${HEADS[head]} — test macro-F1`, pct(t[head].macro_f1),
        `accuracy ${pct(t[head].accuracy)} · F1 female ${pct(s.Female)} · male ${pct(s.Male)} · ${t[head].n} clips`,
        deltaVsBaseline(head + ".macro_f1", t[head].macro_f1, false));
    };
    if (t.displayed_valence) h += classTile("displayed_valence");
    for (const head of ["displayed_intensity", "perceived_intensity"]) if (t[head])
      h += tile(`${HEADS[head]} — test MAE`, num(t[head].mae), `${t[head].n} clips`, deltaVsBaseline(head + ".mae", t[head].mae, true));
    for (const head of ["displayed_emotion", "perceived_emotion"]) if (t[head]) h += classTile(head);
    h += `</div>`;
  } else h += `<p class="empty-note">No test scores: this run did not finish.</p>`;
  h += `<h3>Learning curves</h3>
    <div class="toolbar" role="group" aria-label="Chart controls">
      <span class="legend"><span class="key"><i style="background:var(--train)"></i>Train</span>
      <span class="key"><i style="background:var(--val)"></i>Val</span><span class="key"><i class="best"></i>Best epoch</span></span>
      <span>Epochs <input id="zfrom" type="number" min="1" aria-label="First epoch shown"> to <input id="zto" type="number" min="1" aria-label="Last epoch shown"></span>
      <button type="button" id="zreset">Show all epochs</button>
      <button type="button" id="logbtn" aria-pressed="${state.log}">Log scale for losses</button>
      <span class="hint">Hover for values · drag across a chart to zoom · double-click to reset</span></div>
    <div class="charts" id="charts"></div>
    <details><summary>Loss of each head</summary><p class="note">The total loss is the weighted sum of these.</p><div class="charts" id="losscharts"></div></details>`;
  if (run.test) {
    const cms = run.confusion || {};
    h += `<h3>Test breakdown</h3><div class="grid2">`;
    for (const head of Object.keys(cms)) h += `<div class="scroll">${confusionTable(head, cms[head])}<div style="margin-top:8px">${perClassTable(cms[head])}</div></div>`;
    h += `</div>`;
    const ieo = t.ieo_perceived_intensity_by_level;
    if (ieo) h += `<h3>Intensity check (IEO clips)</h3><p class="note">Mean predicted perceived intensity by the actor's
      intended level; it should rise from LO to HI. Spearman ρ, prediction vs level: <b>${num(ieo.spearman_pred_vs_level, 2)}</b>
      (raters vs level: ${num(ieo.spearman_true_vs_level, 2)}).</p><table class="small" style="width:auto"><thead><tr><th>Level</th>
      ${["LO", "MD", "HI"].map(l => `<th>${l}</th>`).join("")}</tr></thead><tbody><tr><td>Mean prediction</td>
      ${["LO", "MD", "HI"].map(l => `<td>${num(ieo.mean_pred[l])}</td>`).join("")}</tr></tbody></table>`;
  }
  const cols = Object.keys(run.epochs);
  if (cols.length) {
    h += `<details><summary>Every epoch (${n})</summary><div class="scroll"><table class="small epochs"><thead><tr><th>Epoch</th>${cols.map(c => `<th>${esc(c.replaceAll("_", " "))}</th>`).join("")}</tr></thead><tbody>`;
    for (let i = 0; i < n; i++) h += `<tr><td class="${run.best_epoch === i + 1 ? "bestrow" : ""}">${i + 1}${run.best_epoch === i + 1 ? " (best)" : ""}</td>${cols.map(c => {
      const v = run.epochs[c][i]; return `<td class="${run.best_epoch === i + 1 ? "bestrow" : ""}">${v == null ? "–" : c.includes("accuracy") ? pct(v) : num(v, 4)}</td>`; }).join("")}</tr>`;
    h += `</tbody></table></div></details>`;
  }
  el.innerHTML = h;
  const charts = el.querySelector("#charts"), lossCharts = el.querySelector("#losscharts");
  CHARTS.filter(s => run.epochs[s.key]).forEach(s => charts.appendChild(chartBox(s)));
  LOSS_CHARTS.filter(s => run.epochs[s.key]).forEach(s => lossCharts.appendChild(chartBox(s)));
  el.querySelector("details").addEventListener("toggle", redrawCharts);
  el.querySelector("#zreset").addEventListener("click", resetZoom);
  el.querySelector("#logbtn").addEventListener("click", ev => { state.log = !state.log; ev.target.setAttribute("aria-pressed", state.log); redrawCharts(); });
  for (const id of ["zfrom", "zto"]) el.querySelector("#" + id).addEventListener("change", () => {
    let f = +el.querySelector("#zfrom").value || 1, to = +el.querySelector("#zto").value || n;
    f = Math.max(1, Math.min(f, n)); to = Math.max(1, Math.min(to, n)); if (to < f) [f, to] = [to, f];
    state.from = f; state.to = to; redrawCharts();
  });
  redrawCharts();
}

function select(dir, scroll) {
  history.replaceState(null, "", "#" + encodeURIComponent(dir)); render(dir);
  if (scroll) document.getElementById("detail").scrollIntoView({behavior: "smooth", block: "start"});
}
document.querySelectorAll("tr.run").forEach(tr => {
  tr.addEventListener("click", () => select(tr.dataset.run, true));
  tr.addEventListener("keydown", ev => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); select(tr.dataset.run, true); } });
});
new ResizeObserver(() => state.run && redrawCharts()).observe(document.querySelector("main"));
const initial = decodeURIComponent(location.hash.slice(1));
const first = RUNS[initial] ? initial : (DATA.runs.filter(r => r.status === "counted").sort((a, b) => b.dir.localeCompare(a.dir))[0] || {}).dir;
if (first) render(first);
</script>
</body></html>
"""


def baseline(runs: list[dict]) -> dict[str, float]:
    counted = [r for r in runs if r["config"] == B0_CONFIG and r["status"] == "counted"]
    out = {}
    for _, head, name, _ in COLUMNS:
        values = [v for v in (metric(r, head, name) for r in counted) if v is not None]
        if values:
            out[f"{head}.{name}"] = statistics.mean(values)
    return out


def main() -> None:
    runs = load_runs()
    BASELINE.update(baseline(runs))
    data = json.dumps({"runs": runs, "baseline": BASELINE}).replace("</", "<\\/")
    page = (PAGE.replace("__UPDATED__", datetime.now().strftime("%d %b %Y %H:%M"))
            .replace("__SUMMARY__", summary_table(runs)).replace("__RUNS__", runs_table(runs))
            .replace("__DATA__", data))
    out = ROOT / "results.html"
    out.write_text(page)
    done = sum(r["status"] == "counted" for r in runs)
    print(f"{out.name}: {done} counted runs, {len(runs)} run directories")


if __name__ == "__main__":
    main()
