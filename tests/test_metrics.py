"""Metrics on a small hand-made predictions table (no model involved)."""
import numpy as np
import pandas as pd
import pytest

from mffer.metrics import compute


def table():
    return pd.DataFrame({
        "clip_id": ["a", "b", "c", "d"], "actor_id": [1, 1, 2, 2], "sex": ["Male", "Male", "Female", "Female"],
        "sentence": ["IEO", "IEO", "ITS", "IEO"], "displayed_level": ["LO", "HI", "XX", "MD"],
        "displayed_emotion_true": ["ANG", "SAD", "HAP", "NEU"], "displayed_emotion_pred": ["ANG", "SAD", "SAD", "NEU"],
        "displayed_emotion_mask": [1, 1, 1, 1],
        "perceived_emotion_true": ["ANG", "", "HAP", "NEU"], "perceived_emotion_pred": ["SAD", "SAD", "HAP", "NEU"],
        "perceived_emotion_mask": [1, 0, 1, 1],
        "displayed_intensity_true": [1 / 3, 1.0, np.nan, 2 / 3], "displayed_intensity_pred": [0.5, 0.5, 0.9, 0.5],
        "displayed_intensity_mask": [1, 1, 0, 1],
        "perceived_intensity_true": [0.5, 0.8, 0.6, 0.6], "perceived_intensity_pred": [0.4, 0.7, 0.6, 0.6],
        "perceived_intensity_mask": [1, 1, 1, 1],
    })


def test_accuracy_ignores_masked_clips_and_splits_by_sex():
    m = compute(table())
    assert m["displayed_emotion"]["accuracy"] == pytest.approx(0.75)
    assert m["displayed_emotion"]["accuracy_by_sex"] == {"Female": 0.5, "Male": 1.0}
    assert m["perceived_emotion"]["n"] == 3 and m["perceived_emotion"]["accuracy"] == pytest.approx(2 / 3)


def test_mae_only_over_present_labels():
    m = compute(table())
    assert m["displayed_intensity"]["n"] == 3
    assert m["displayed_intensity"]["mae"] == pytest.approx((1 / 6 + 0.5 + 1 / 6) / 3)
    assert m["perceived_intensity"]["mae"] == pytest.approx(0.05)


def test_ieo_ordering_check_uses_levelled_clips_only():
    m = compute(table())["ieo_perceived_intensity_by_level"]
    assert set(m["mean_pred"]) == {"LO", "MD", "HI"}
    assert m["spearman_pred_vs_level"] == pytest.approx(1.0)
