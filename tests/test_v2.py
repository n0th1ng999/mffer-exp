"""V2 invariants (ADRs 0024-0026): pooled splits, the corpus partition, model B's loss and shapes, valence read off
the emotion for single-task models, and protocol 4's selection score."""
import json

import numpy as np
import pandas as pd
import pytest

from mffer.config import ROOT, load_paths
from mffer.data.cremad import VALENCE
from mffer.data.splits import FRACTIONS, load_split, partition, rows
from mffer.metrics import predictions_table, score
from mffer.models import EMBED_DIM, MOCANU_AUDIO, MOCANU_VIDEO, POLARITY, PRESETS, EmotionMetricLoss, build_model

V2_CLIPS = load_paths().processed / "v2" / "clips.csv"
needs_v2 = pytest.mark.skipif(not V2_CLIPS.exists(), reason="run `python -m mffer.data.v2 index` first")


@pytest.fixture(scope="module")
def clips():
    return pd.read_csv(V2_CLIPS)


@needs_v2
@pytest.mark.parametrize("split_id", [0, 1, 2])
def test_v2_split_keeps_v1_cremad_actors_and_is_actor_disjoint(clips, split_id):
    v1 = json.loads((ROOT / "splits" / f"split_{split_id}.json").read_text())
    v2 = load_split(split_id, "v2")
    for part in FRACTIONS:
        assert sorted(a for a in v2[part] if a < 2000) == sorted(v1[part])        # CREMA-D exactly as V1
    parts = [set(v2[p]) for p in FRACTIONS]
    assert set.union(*parts) == set(clips.actor_id)                               # every actor, once
    ravdess = {p: len([a for a in v2[p] if a > 2000]) for p in FRACTIONS}
    assert ravdess == {"train": 18, "val": 2, "test": 4}
    for part in FRACTIONS:   # both corpora and both sexes in every part
        sub = clips[clips.actor_id.isin(v2[part])]
        assert set(sub.corpus) == {"cremad", "ravdess"} and set(sub.sex) == {"Female", "Male"}


@needs_v2
def test_corpus_partition_gives_one_client_per_corpus(clips):
    train = rows(clips, 0, "train", "v2")
    clients = partition(clips, train, 2, "corpus", seed=0)
    assert [set(clips.corpus.iloc[c]) for c in clients] == [{"cremad"}, {"ravdess"}]
    assert sum(len(c) for c in clients) == len(train)
    with pytest.raises(AssertionError):
        partition(clips, train, 3, "corpus", seed=0)


@needs_v2
def test_ravdess_labels_follow_the_cremad_schema(clips):
    rav = clips[clips.corpus == "ravdess"]
    assert (rav.displayed_valence == rav.displayed_emotion.map(VALENCE)).all()
    assert rav.perceived_emotion.isna().all() and rav.perceived_intensity.isna().all()   # masked heads
    assert rav.loc[rav.displayed_emotion == "NEU", "displayed_intensity"].isna().all()
    assert set(rav.displayed_intensity.dropna().round(3)) == {0.667, 1.0}


def test_polarity_groups_are_the_valence_classes():
    emotions = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
    by_valence = {}
    for emotion, group in zip(emotions, POLARITY, strict=True):
        by_valence.setdefault(VALENCE[emotion], set()).add(group)
    assert all(len(groups) == 1 for groups in by_valence.values())
    assert len({g for gs in by_valence.values() for g in gs}) == 3


def test_emotion_metric_loss_is_zero_when_separated_and_positive_when_not():
    loss = EmotionMetricLoss()
    y = np.eye(6, dtype="float32")[[0, 0, 1, 1, 3, 3, 4, 4]]   # ANG ANG DIS DIS HAP HAP NEU NEU
    good = np.array([[1, 0, 0], [1, 0, 0], [.9, .436, 0], [.9, .436, 0], [0, 0, 1], [0, 0, 1], [-1, 0, 0], [-1, 0, 0]],
                    "float32")
    good /= np.linalg.norm(good, axis=1, keepdims=True)
    assert np.allclose(np.asarray(loss.call(y, good)), 0)
    # anger anchor placed next to happiness (other polarity) rather than disgust (related): penalised
    bad = good.copy()
    bad[0] = good[4]
    assert np.asarray(loss.call(y, bad))[0] > 0


def test_model_b_shapes_and_frozen_encoders_are_outside_the_model():
    model = build_model(PRESETS["mocanu"])
    x = {"mocanu_video": np.zeros((2, *MOCANU_VIDEO), "float32"), "mocanu_audio": np.zeros((2, *MOCANU_AUDIO), "float32")}
    out = model(x)
    assert tuple(out["displayed_emotion"].shape) == (2, 6) and tuple(out["emotion_embedding"].shape) == (2, EMBED_DIM)
    assert model.count_params() < 20_000_000   # attention + fusion + classifier only (ADR 0025)


def test_single_task_model_gets_valence_from_its_emotion_and_the_score_averages_corpora():
    clips = pd.DataFrame({"clip_id": list("abcdef"), "actor_id": [1, 1, 2, 2, 3, 3], "sex": ["Male"] * 6,
                          "sentence": ["X"] * 6, "displayed_level": ["XX"] * 6,
                          "corpus": ["cremad"] * 4 + ["ravdess"] * 2})
    true = ["HAP", "SAD", "NEU", "ANG", "HAP", "FEA"]
    emotions = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
    valences = ["negative", "neutral", "positive"]
    targets = {"displayed_emotion": np.eye(6, dtype="float32")[[emotions.index(e) for e in true]],
               "displayed_valence": np.eye(3, dtype="float32")[[valences.index(VALENCE[e]) for e in true]]}
    weights = {h: np.ones(6, "float32") for h in targets}
    predicted = ["HAP", "ANG", "NEU", "SAD", "NEU", "FEA"]   # wrong emotion but right valence on clips b and d
    outputs = {"displayed_emotion": np.eye(6, dtype="float32")[[emotions.index(e) for e in predicted]],
               "emotion_embedding": np.zeros((6, EMBED_DIM), "float32")}
    table = predictions_table(clips, np.arange(6), targets, weights, outputs)
    assert list(table.displayed_valence_pred) == [VALENCE[e] for e in predicted]
    parts = score(table)
    expected = np.mean([np.mean([parts["cremad_displayed_valence_macro_f1"], parts["ravdess_displayed_valence_macro_f1"]]),
                        np.mean([parts["cremad_displayed_emotion_macro_f1"], parts["ravdess_displayed_emotion_macro_f1"]])])
    assert parts["score"] == pytest.approx(expected)
