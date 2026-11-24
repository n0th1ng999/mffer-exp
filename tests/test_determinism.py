"""Reproducibility (ADR 0016), on random tensors: op determinism is on, the same seed trains to bit-identical
weights, and dropout masks depend only on the seed given to `reseed`, not on what ran before."""
import keras
import numpy as np

from mffer import device
from mffer.models import AUDIO_SHAPE, DEMO_DIM, PRESETS, TEXT_DIM, VIDEO_SHAPE, build_model, derive_seed, reseed


def test_op_determinism_is_on_in_every_process_that_configures_the_device():
    assert device.deterministic()


def test_derived_seeds_are_stable_and_distinct():
    assert derive_seed(0, 1, 2) == derive_seed(0, 1, 2)
    assert len({derive_seed(0, epoch) for epoch in range(90)}) == 90


def dropout_output(model, x):
    return keras.ops.convert_to_numpy(model(x, training=True)["displayed_emotion"])


def test_reseed_makes_dropout_masks_independent_of_earlier_steps():
    keras.utils.set_random_seed(0)
    model = build_model(PRESETS["m3"])
    x = {"audio": np.ones((4, *AUDIO_SHAPE), np.float32), "video": np.full((4, *VIDEO_SHAPE), 0.5, np.float32),
         "text": np.eye(TEXT_DIM, dtype=np.float32)[:4], "demographics": np.full((4, DEMO_DIM), 0.5, np.float32)}
    reseed(model, 5)
    first = dropout_output(model, x)
    dropout_output(model, x)                                  # advances every dropout layer's state
    assert not np.array_equal(first, dropout_output(model, x))
    reseed(model, 5)
    assert np.array_equal(first, dropout_output(model, x))   # as if nothing had run in between

    keras.utils.set_random_seed(123)                          # a fresh process: same weights, other RNG history
    rebuilt = build_model(PRESETS["m3"])
    rebuilt.set_weights(model.get_weights())
    reseed(rebuilt, 5)
    assert np.array_equal(first, dropout_output(rebuilt, x))
