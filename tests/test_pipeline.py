"""Invariants of the splits and the input pipeline, checked on the real prepared data."""
import numpy as np
import pytest

from mffer.data.pipeline import AudioStats, Cremad, fit_stats, label_coverage, make_dataset
from mffer.data.splits import partition, rows
from mffer.metrics import MONITOR
from mffer.models import PRESETS, build_model


@pytest.fixture(scope="module")
def data():
    return Cremad()


@pytest.mark.parametrize("split_id", [0, 1, 2])
def test_split_parts_are_disjoint_by_actor_and_cover_all_clips(data, split_id):
    parts = {p: rows(data.clips, split_id, p) for p in ("train", "val", "test")}
    actors = {p: set(data.clips.actor_id.iloc[r]) for p, r in parts.items()}
    assert not actors["train"] & actors["val"] and not actors["train"] & actors["test"] and not actors["val"] & actors["test"]
    assert sum(len(r) for r in parts.values()) == len(data.clips)


@pytest.mark.parametrize("mode", ["iid", "actor"])
def test_partition_gives_each_training_clip_to_exactly_one_client(data, mode):
    train = rows(data.clips, 0, "train")
    clients = partition(data.clips, train, n_clients=2, mode=mode, seed=0)
    assert np.array_equal(np.sort(np.concatenate(clients)), np.sort(train))
    if mode == "actor":
        a, b = (set(data.clips.actor_id.iloc[c]) for c in clients)
        assert not a & b


def test_masks_match_the_data_decisions(data):
    w = data.weights
    assert (w["perceived_emotion"] == 0).sum() == 420
    assert (w["displayed_intensity"] == 1).sum() == 1365
    ieo_levelled = (data.clips.sentence == "IEO") & data.clips.displayed_level.isin(["LO", "MD", "HI"])
    assert np.array_equal(w["displayed_intensity"] == 1, ieo_levelled.to_numpy())
    for head in ("displayed_emotion", "displayed_valence", "perceived_intensity"):
        assert (w[head] == 1).all()
    assert (data.targets["perceived_emotion"][w["perceived_emotion"] == 0] == 0).all()


def test_audio_normalised_with_train_stats_and_padding_kept_at_zero(data):
    train = rows(data.clips, 0, "train")
    stats = fit_stats(data, train)
    sample = train[::13]   # spread over all training actors (consecutive rows are one actor)
    batch = next(iter(make_dataset(data, sample, PRESETS["m1"], stats, training=False, batch_size=len(sample))))[0]
    audio = batch["audio"].numpy()
    frames = data.read(sample, video=False)["audio_frames"]
    valid = np.arange(100)[None, :] < frames[:, None]
    assert (audio[~valid] == 0).all()
    assert np.allclose(audio[valid].mean(0), 0, atol=0.25) and np.allclose(audio[valid].std(0), 1, atol=0.25)


def test_combined_client_stats_equal_stats_of_all_their_data(data):
    train = rows(data.clips, 0, "train")
    clients = partition(data.clips, train, n_clients=2, mode="actor", seed=0)
    pooled = AudioStats.combine([fit_stats(data, c) for c in clients])
    direct = fit_stats(data, train)
    assert np.allclose(pooled.mean, direct.mean, atol=1e-3) and np.allclose(pooled.std, direct.std, atol=1e-3)


def test_evaluation_keeps_row_order_and_inputs_follow_the_spec(data):
    test = rows(data.clips, 0, "test")[:40]
    stats = fit_stats(data, rows(data.clips, 0, "train"))
    batches = list(make_dataset(data, test, PRESETS["m3"], stats, training=False))
    x, y, w = batches[0]
    assert set(x) == {"audio", "video", "text", "demographics"}
    assert x["video"].dtype.name == "float32" and float(x["video"].numpy().max()) <= 1.0
    got = np.concatenate([b[1]["displayed_emotion"].numpy() for b in batches])
    assert np.array_equal(got, data.targets["displayed_emotion"][test])
    assert [len(b[1]["displayed_emotion"]) for b in batches] == [16, 16, 8]


def test_training_order_is_reshuffled_each_epoch_but_reproducible(data):
    some = rows(data.clips, 0, "train")[:64]
    stats = fit_stats(data, some)
    def first(ds):
        return next(iter(ds))[1]["displayed_emotion"].numpy()

    ds = make_dataset(data, some, PRESETS["m1"], stats, training=True, seed=7)
    epoch0, epoch1 = first(ds), first(ds)
    again = first(make_dataset(data, some, PRESETS["m1"], stats, training=True, seed=7))
    assert np.array_equal(epoch0, again) and not np.array_equal(epoch0, epoch1)


def test_a_resumed_dataset_continues_the_shuffle_of_an_uninterrupted_one(data):
    some = rows(data.clips, 0, "train")[:64]
    stats = fit_stats(data, some)
    def first(ds):
        return next(iter(ds))[1]["displayed_emotion"].numpy()

    uninterrupted = make_dataset(data, some, PRESETS["m1"], stats, training=True, seed=7)
    epochs = [first(uninterrupted) for _ in range(3)]
    resumed = make_dataset(data, some, PRESETS["m1"], stats, training=True, seed=7, first_epoch=2)
    assert np.array_equal(first(resumed), epochs[2])


def test_shuffles_of_neighbouring_seeds_do_not_repeat_shifted_by_one_epoch(data):
    """Shuffle seeds come from derive_seed(seed, epoch), not seed + epoch (ADR 0016): seed 1's epoch 0
    must not replay seed 0's epoch 1."""
    some = rows(data.clips, 0, "train")[:64]
    stats = fit_stats(data, some)
    def order(seed, epoch):
        ds = make_dataset(data, some, PRESETS["audio_only"], stats, training=True, seed=seed, first_epoch=epoch,
                          batch_size=len(some))
        return next(iter(ds))[1]["displayed_emotion"].numpy()

    assert not np.array_equal(order(1, 0), order(0, 1))


def test_keras_trains_on_the_dataset_with_masked_heads(data):
    train = rows(data.clips, 0, "train")[:32]
    stats = fit_stats(data, train)
    spec = PRESETS["m1"]
    model = build_model(spec, coverage=label_coverage(data, rows(data.clips, 0, "train"), spec))
    history = model.fit(make_dataset(data, train, spec, stats, training=True), epochs=1, verbose=0)
    assert np.isfinite(history.history["loss"][0])
    assert "displayed_intensity_mae" in history.history
    assert MONITOR.removeprefix("val_") in history.history   # the early-stopping metric is logged


@pytest.mark.parametrize("split_id", [0, 1, 2])
def test_local_only_client_holds_the_rows_of_the_same_federated_client(data, split_id):
    """L1 client k (mffer.train --client k) and F2 client k (mffer.federated) must train on the same rows."""
    from mffer.data.splits import client_rows
    train = rows(data.clips, split_id, "train")
    federated = partition(data.clips, train, n_clients=2, mode="actor", seed=split_id)
    local = client_rows(data.clips, split_id, 2, "actor")
    assert all(np.array_equal(fed, loc) for fed, loc in zip(federated, local, strict=True))
