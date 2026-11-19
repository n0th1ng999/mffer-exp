"""Metrics computed from a predictions table (one row per clip). Same code for every training mode.

A predictions table has the clip metadata (clip_id, actor_id, sex, sentence, displayed_level) plus, per head,
`<head>_true` / `<head>_pred` (class name or intensity) and `<head>_mask` (1 = label present).
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

from mffer.data.cremad import EMOTIONS, VALENCES

CLASSES = {"displayed_emotion": EMOTIONS, "perceived_emotion": EMOTIONS, "displayed_valence": VALENCES}
INTENSITIES = ["displayed_intensity", "perceived_intensity"]

# The paper's metrics in priority order (ADR 0014): valence macro-F1 is primary (did the customer react
# well?); macro, because always answering "negative" already scores 68% accuracy. Then the two intensity MAEs.
PRIORITY = ["displayed_valence.macro_f1", "displayed_intensity.mae", "perceived_intensity.mae"]
# Its validation version picks the epoch (or round) in every training mode; logged by models.compile_model.
MONITOR = "val_displayed_valence_macro_f1"


def macro_f1(true: pd.Series, pred: pd.Series, labels: list[str]) -> float:
    return float(f1_score(true, pred, labels=labels, average="macro", zero_division=0))


def predictions_table(clips: pd.DataFrame, rows: np.ndarray, targets: dict, weights: dict,
                      outputs: dict[str, np.ndarray]) -> pd.DataFrame:
    """Join model outputs (in `rows` order) with labels and metadata."""
    table = clips.iloc[rows][["clip_id", "actor_id", "sex", "sentence", "displayed_level"]].reset_index(drop=True)
    for head, out in outputs.items():
        table[f"{head}_mask"] = weights[head][rows].astype(int)
        if head in CLASSES:
            names = np.array(CLASSES[head])
            table[f"{head}_true"] = np.where(table[f"{head}_mask"] == 1, names[targets[head][rows].argmax(1)], "")
            table[f"{head}_pred"] = names[out.argmax(1)]
        else:
            table[f"{head}_true"] = np.where(table[f"{head}_mask"] == 1, targets[head][rows][:, 0], np.nan)
            table[f"{head}_pred"] = out[:, 0]
    return table


def compute(table: pd.DataFrame) -> dict:
    """Accuracy and macro-F1 per classification head (overall and by sex), MAE per intensity head, and
    whether predicted perceived intensity rises with the displayed level on IEO clips."""
    result: dict = {"n_clips": len(table), "n_actors": int(table.actor_id.nunique())}
    for head in CLASSES:
        if f"{head}_pred" not in table:
            continue
        scored = table[table[f"{head}_mask"] == 1]
        correct = scored[f"{head}_true"] == scored[f"{head}_pred"]
        result[head] = {
            "n": len(scored),
            "accuracy": float(correct.mean()),
            "macro_f1": macro_f1(scored[f"{head}_true"], scored[f"{head}_pred"], CLASSES[head]),
            "accuracy_by_sex": {sex: float(c.mean()) for sex, c in correct.groupby(scored["sex"])},
            "macro_f1_by_sex": {sex: macro_f1(g[f"{head}_true"], g[f"{head}_pred"], CLASSES[head])
                                for sex, g in scored.groupby("sex")},
        }
    for head in INTENSITIES:
        if f"{head}_pred" not in table:
            continue
        scored = table[table[f"{head}_mask"] == 1]
        result[head] = {"n": len(scored),
                        "mae": float((scored[f"{head}_true"] - scored[f"{head}_pred"]).abs().mean())}
    if "perceived_intensity_pred" in table:
        ieo = table[table.displayed_level.isin(["LO", "MD", "HI"])]
        if len(ieo):
            level = ieo.displayed_level.map({"LO": 1, "MD": 2, "HI": 3})

            def rho(x):   # undefined (None) for a constant predictor such as B0, not NaN: keeps JSON valid
                return None if np.ptp(x) == 0 else float(spearmanr(x, level)[0])
            result["ieo_perceived_intensity_by_level"] = {
                "mean_pred": {lv: float(g.mean()) for lv, g in ieo.groupby("displayed_level")["perceived_intensity_pred"]},
                "spearman_pred_vs_level": rho(ieo.perceived_intensity_pred),
                "spearman_true_vs_level": rho(ieo.perceived_intensity_true),
            }
    return result
