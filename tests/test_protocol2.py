"""Protocol-2 options (ADRs 0018-0022) on the real prepared data: masked audio pooling, valence class
weights, log-mel audio, augmentation, capacity flags. All of them are training-side; evaluation is unchanged."""
from dataclasses import replace

import keras
import numpy as np
import pytest

from mffer.data.pipeline import Cremad, fit_stats, label_coverage, make_dataset, valence_class_weights
from mffer.data.splits import client_rows, rows
from mffer.metrics import MONITOR
from mffer.models import PRESETS, MaskedMean, build_model


@pytest.fixture(scope="module")
def data():
    return Cremad()


def spec(preset, **flags):
    return replace(PRESETS[preset], **flags)


def first_batch(ds):
    return next(iter(ds))


# ---------------------------------------------------------------- masked audio pooling (ADR 0018)

@pytest.mark.parametrize("features", ["mfcc", "logmel"])
def test_masked_audio_encoding_does_not_depend_on_padding(features):
    model = build_model(spec("audio_only", audio_pool="masked", audio_features=features))
    pooled = next(layer for layer in model.layers if isinstance(layer, MaskedMean))
    encoder = keras.Model(model.input, pooled.output)
    frames, bands = model.input["audio"].shape[1:] if isinstance(model.input, dict) else model.input.shape[1:]
    clip = np.random.default_rng(0).standard_normal((frames // 2, bands)).astype(np.float32)
    short, long_ = np.zeros((2, frames, bands), np.float32), np.zeros((2, frames, bands), np.float32)
    short[0, :frames // 2] = clip                          # clip then padding
    long_[1, :frames // 2] = clip                          # the same clip in another batch slot
    long_[0, :frames // 4] = clip[:frames // 4]            # a shorter neighbour, to vary the batch
    a = encoder.predict({"audio": short}, verbose=0)[0]
    b = encoder.predict({"audio": long_}, verbose=0)[1]
    assert np.allclose(a, b, atol=1e-5)


# ---------------------------------------------------------------- valence class weights (ADR 0019)

def test_valence_weights_average_one_on_the_training_rows_and_favour_rare_classes(data):
    train = rows(data.clips, 0, "train")
    w = np.array(valence_class_weights(data, train))
    classes = data.targets["displayed_valence"][train].argmax(1)
    assert np.isclose(w[classes].mean(), 1.0)
    assert w[0] < 1 < min(w[1], w[2])                     # VALENCES: negative, neutral, positive


def test_valence_weights_apply_to_training_only(data):
    train = rows(data.clips, 0, "train")[::7]
    stats = fit_stats(data, train)
    s = spec("audio_only", balance_valence=True)
    w_train = first_batch(make_dataset(data, train, s, stats, training=True, batch_size=len(train)))[2]
    w_eval = first_batch(make_dataset(data, train, s, stats, training=False, batch_size=len(train)))[2]
    assert len(np.unique(w_train["displayed_valence"].numpy())) == 3
    assert set(np.unique(w_eval["displayed_valence"].numpy())) == {1.0}
    assert np.array_equal(w_train["displayed_emotion"].numpy(), w_eval["displayed_emotion"].numpy())


def test_federated_clients_weight_with_their_own_rows(data):
    clients = client_rows(data.clips, 0, 2, "actor")
    assert valence_class_weights(data, clients[0]) != valence_class_weights(data, clients[1])


# ---------------------------------------------------------------- log-mel audio (ADR 0020)

def test_logmel_rows_follow_the_clips_and_padding_stays_zero(data):
    train = rows(data.clips, 0, "train")
    stats = fit_stats(data, train, "logmel")
    assert len(stats.mean) == 64
    sample = train[::13]
    s = spec("audio_only", audio_features="logmel")
    audio = first_batch(make_dataset(data, sample, s, stats, training=False, batch_size=len(sample)))[0]["audio"]
    frames = data.read(sample, video=False, features="logmel")["audio_frames"]
    valid = np.arange(400)[None, :] < frames[:, None]
    assert audio.shape == (len(sample), 400, 64)
    assert (audio.numpy()[~valid] == 0).all()
    assert np.allclose(audio.numpy()[valid].mean(0), 0, atol=0.25)


# ---------------------------------------------------------------- augmentation (ADR 0021)

@pytest.mark.parametrize("features", ["mfcc", "logmel"])
def test_augmentation_is_reproducible_per_epoch_and_training_only(data, features):
    some = rows(data.clips, 0, "train")[:16]
    stats = fit_stats(data, some, features)
    s = spec("m1", augment=True, audio_features=features)
    plain = spec("m1", audio_features=features)

    def batch(sp, training, epoch=0):
        x = first_batch(make_dataset(data, some, sp, stats, training=training, seed=3, first_epoch=epoch))[0]
        return {k: v.numpy() for k, v in x.items()}

    a, again, other = batch(s, True), batch(s, True), batch(s, True, epoch=1)
    assert all(np.array_equal(a[k], again[k]) for k in a)
    assert not np.array_equal(a["video"], batch(plain, True)["video"])        # same order, augmented
    assert not np.array_equal(a["video"], other["video"])
    evaluated, plain_eval = batch(s, False), batch(plain, False)
    assert all(np.array_equal(evaluated[k], plain_eval[k]) for k in evaluated)  # evaluation untouched
    if features == "logmel":
        assert not np.array_equal(a["audio"], batch(plain, True)["audio"])
    else:
        assert np.array_equal(a["audio"], batch(plain, True)["audio"])         # MFCC never masked


# ---------------------------------------------------------------- everything together trains

def test_full_protocol_2_model_trains_and_logs_the_monitor(data):
    train = rows(data.clips, 0, "train")[:32]
    s = spec("m3", audio_pool="masked", audio_features="logmel", balance_valence=True, augment=True,
             width=2, visual_blocks=4)
    stats = fit_stats(data, train, "logmel")
    model = build_model(s, coverage=label_coverage(data, rows(data.clips, 0, "train"), s))
    history = model.fit(make_dataset(data, train, s, stats, training=True), epochs=1, verbose=0,
                        validation_data=make_dataset(data, train, s, stats, training=False))
    assert np.isfinite(history.history["loss"][0]) and MONITOR in history.history
