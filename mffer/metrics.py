"""Metrics computed from a predictions table (one row per clip). Same code for every training mode.

A predictions table has the clip metadata (clip_id, actor_id, sex, sentence, displayed_level) plus, per head,
`<head>_true` / `<head>_pred` (class name or intensity) and `<head>_mask` (1 = label present).
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

from mffer.data.cremad import EMOTIONS, VALENCE, VALENCES

CLASSES = {"displayed_emotion": EMOTIONS, "perceived_emotion": EMOTIONS, "displayed_valence": VALENCES}
INTENSITIES = ["displayed_intensity", "perceived_intensity"]

# The paper's metrics in priority order (ADR 0014): valence macro-F1 is primary (did the customer react
# well?); macro, because always answering "negative" already scores 68% accuracy. Then the two intensity MAEs.
PRIORITY = ["displayed_valence.macro_f1", "displayed_intensity.mae", "perceived_intensity.mae"]
# Its validation version picked the epoch (or round) in protocols 1-3; logged by models.compile_model.
MONITOR = "val_displayed_valence_macro_f1"
# Protocol 4 (ADR 0026): the epoch, the round and the T1 winner are chosen by SCORE, the mean of valence and
# emotion macro-F1, each averaged over the corpora (CREMA-D, RAVDESS) so the 7x larger corpus doesn't decide it.
# Logged by train.ValScore and federated.RoundLog.
SCORE = "val_score"
SCORED_HEADS = ("displayed_valence", "displayed_emotion")


def macro_f1(true: pd.Series, pred: pd.Series, labels: list[str]) -> float:
    return float(f1_score(true, pred, labels=labels, average="macro", zero_division=0))


def predictions_table(clips: pd.DataFrame, rows: np.ndarray, targets: dict, weights: dict,
                      outputs: dict[str, np.ndarray]) -> pd.DataFrame:
    """Join model outputs (in `rows` order) with labels and metadata."""
    columns = ["clip_id", "actor_id", "sex", "sentence", "displayed_level"] + (["corpus"] if "corpus" in clips else [])
    table = clips.iloc[rows][columns].reset_index(drop=True)
    outputs = {h: o for h, o in outputs.items() if h in CLASSES or h in INTENSITIES}   # not model B's embedding
    for head, out in outputs.items():
        table[f"{head}_mask"] = weights[head][rows].astype(int)
        if head in CLASSES:
            names = np.array(CLASSES[head])
            table[f"{head}_true"] = np.where(table[f"{head}_mask"] == 1, names[targets[head][rows].argmax(1)], "")
            table[f"{head}_pred"] = names[out.argmax(1)]
        else:
            table[f"{head}_true"] = np.where(table[f"{head}_mask"] == 1, targets[head][rows][:, 0], np.nan)
            table[f"{head}_pred"] = out[:, 0]
    if "displayed_valence" not in outputs and "displayed_emotion" in outputs:
        # single-task emotion model (B): valence is a function of the emotion (mffer.data.cremad.VALENCE)
        head = "displayed_valence"
        names = np.array(VALENCES)
        table[f"{head}_mask"] = weights[head][rows].astype(int)
        table[f"{head}_true"] = np.where(table[f"{head}_mask"] == 1, names[targets[head][rows].argmax(1)], "")
        table[f"{head}_pred"] = table["displayed_emotion_pred"].map(VALENCE)
        table["valence_from_emotion"] = 1
    return table


def score(table: pd.DataFrame) -> dict[str, float]:
    """Protocol 4's selection score and its parts: macro-F1 of each SCORED_HEADS head on each corpus, and
    `score` = the mean over heads of the mean over corpora."""
    corpora = table["corpus"] if "corpus" in table else pd.Series("cremad", index=table.index)
    parts: dict[str, float] = {}
    for head in SCORED_HEADS:
        for corpus, group in table.groupby(corpora):
            scored = group[group[f"{head}_mask"] == 1]
            parts[f"{corpus}_{head}_macro_f1"] = macro_f1(scored[f"{head}_true"], scored[f"{head}_pred"], CLASSES[head])
    per_head = [np.mean([v for k, v in parts.items() if k.endswith(f"_{head}_macro_f1")]) for head in SCORED_HEADS]
    return {"score": float(np.mean(per_head)), **parts}


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
        if "corpus" in scored:
            result[head]["macro_f1_by_corpus"] = {c: macro_f1(g[f"{head}_true"], g[f"{head}_pred"], CLASSES[head])
                                                  for c, g in scored.groupby("corpus")}
            result[head]["accuracy_by_corpus"] = {c: float(k.mean()) for c, k in correct.groupby(scored["corpus"])}
    for head in INTENSITIES:
        if f"{head}_pred" not in table:
            continue
        scored = table[table[f"{head}_mask"] == 1]
        result[head] = {"n": len(scored),
                        "mae": float((scored[f"{head}_true"] - scored[f"{head}_pred"]).abs().mean())}
    if {"displayed_valence_pred", "displayed_emotion_pred"} <= set(table):
        result["score"] = score(table)
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
