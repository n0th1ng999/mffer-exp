"""FedProx client model (random tensors, small model): with mu = 0 it trains exactly like the plain model,
with mu > 0 it stays closer to the global weights, and evaluation never sees the proximal term."""
import keras
import numpy as np
import pytest

from mffer.federated import ProxModel
from mffer.models import AUDIO_SHAPE, HEADS, PRESETS, build_model, compile_model

SPEC = PRESETS["audio_only"]


def batch(n=32):
    rng = np.random.default_rng(0)
    x = {"audio": rng.standard_normal((n, *AUDIO_SHAPE), dtype=np.float32)}
    y = {h: (np.eye(u, dtype=np.float32)[rng.integers(0, u, n)] if u > 1 else rng.random((n, 1), dtype=np.float32))
         for h, (u, _, _) in HEADS.items() if h in SPEC.heads}
    return x, y


def trained(mu: float | None, steps=20):
    """Weights after `steps` updates from a fixed start; mu None = plain model, else ProxModel."""
    keras.utils.set_random_seed(0)
    model = build_model(SPEC)
    if mu is not None:
        model = ProxModel(model.input, model.output)
    start = [w.copy() for w in model.get_weights()]
    compile_model(model, SPEC)
    if mu is not None:
        model.set_anchor(mu, [w.numpy() for w in model.trainable_weights])
    x, y = batch()
    keras.utils.set_random_seed(1)
    model.fit(x, y, epochs=steps, batch_size=32, verbose=0, shuffle=False)
    return model, start


def distance(a, b):
    return sum(float(((u - v) ** 2).sum()) for u, v in zip(a, b, strict=True))


def test_mu_zero_matches_plain_training_and_keeps_the_weight_list():
    plain, _ = trained(None)
    prox, _ = trained(0.0)
    assert len(prox.get_weights()) == len(plain.get_weights())   # FedAvg exchanges the same arrays
    assert all(np.array_equal(a, b) for a, b in zip(plain.get_weights(), prox.get_weights(), strict=True))


def test_larger_mu_stays_closer_to_the_global_weights():
    drift = {mu: distance(*(lambda m, s: (m.get_weights(), s))(*trained(mu))) for mu in (0.0, 1.0, 10.0)}
    assert drift[10.0] < drift[1.0] < drift[0.0]


def test_evaluation_loss_has_no_proximal_term():
    model, _ = trained(10.0)
    x, y = batch()
    with_prox = model.evaluate(x, y, verbose=0, return_dict=True)["loss"]
    model.set_anchor(0.0, [w.numpy() for w in model.trainable_weights])
    assert model.evaluate(x, y, verbose=0, return_dict=True)["loss"] == pytest.approx(with_prox)


def test_reset_optimizer_trains_exactly_like_a_freshly_compiled_model():
    """Federated clients reuse one compiled model and reset Adam each round instead of recompiling (which
    leaks memory): two rounds must give bit-identical weights either way."""
    from mffer.models import reset_optimizer
    x, y = batch()
    start = build_model(SPEC).get_weights()

    def rounds(reuse: bool):
        keras.utils.set_random_seed(0)
        model = build_model(SPEC)
        weights = start
        for _ in range(2):
            model.set_weights(weights)
            if reuse:
                reset_optimizer(model)
            else:
                compile_model(model, SPEC)
            keras.utils.set_random_seed(1)
            model.fit(x, y, batch_size=8, epochs=1, verbose=0, shuffle=False)
            weights = model.get_weights()
        return weights

    for a, b in zip(rounds(reuse=True), rounds(reuse=False), strict=True):
        np.testing.assert_array_equal(a, b)
