"""One model builder for every configuration (M1, the design-ladder steps, M3).

Configurations differ only by the flags in `ModelSpec`, so a step of the ladder changes exactly what its
flags say. Layer sizes follow Table 1 of the paper draft.
"""
from dataclasses import dataclass, field, replace

import keras
import numpy as np
from keras import layers

N_EMOTIONS = 6
N_VALENCE = 3
AUDIO_SHAPES = {                   # frames x coefficients, per audio feature (mffer.data.features)
    "mfcc": (100, 13),             # 13 MFCCs, 40 ms hop, 4 s
    "logmel": (400, 64),           # 64 log-mel bands, 10 ms hop, 4 s (ADR 0020)
}
AUDIO_SHAPE = AUDIO_SHAPES["mfcc"]
VIDEO_SHAPE = (10, 224, 224, 3)    # frames x height x width x RGB
TEXT_DIM = 12                      # one-hot sentence ID (CREMA-D has 12 scripted sentences)
DEMO_DIM = 2                       # scaled age, sex

HEADS = {  # name: (units, activation, loss)
    "displayed_emotion": (N_EMOTIONS, "softmax", "categorical_crossentropy"),
    "perceived_emotion": (N_EMOTIONS, "softmax", "categorical_crossentropy"),
    "displayed_valence": (N_VALENCE, "softmax", "categorical_crossentropy"),
    "displayed_intensity": (1, "sigmoid", "mse"),
    "perceived_intensity": (1, "sigmoid", "mse"),
}


@dataclass(frozen=True)
class ModelSpec:
    audio: bool = True
    video: bool = True
    visual: str = "simple"          # "simple" (M1) or "deep" (3 x Conv2D + BN, BiLSTM)
    gating: bool = False            # scalar sigmoid gate on each modality encoding
    text: bool = False
    demographics: bool = False
    # Protocol 2 options (ADRs 0018-0022). Defaults keep the protocol-1 model; dev runs switch them on.
    audio_pool: str = "mean"        # "mean" over all 100 frames, or "masked": real (non-padded) frames only
    audio_features: str = "mfcc"    # "mfcc" (13 x 40 ms) or "logmel" (64 bands x 10 ms, 2-D conv encoder)
    balance_valence: bool = False   # training-only valence class weights (pipeline.valence_class_weights)
    augment: bool = False           # training-only video flip/crop and log-mel time/frequency masks
    width: int = 1                  # multiplies the dense / LSTM sizes after the encoders
    visual_blocks: int = 3          # conv blocks of the deep visual encoder (3: 32-128 filters, 4: 32-256)
    loss_weights: dict = field(default_factory=lambda: {
        "displayed_emotion": 1.0, "perceived_emotion": 1.0, "displayed_valence": 1.0,
        "displayed_intensity": 0.2, "perceived_intensity": 0.2,
    })

    @property
    def heads(self) -> list[str]:
        return list(self.loss_weights)

    @property
    def audio_shape(self) -> tuple[int, int]:
        return AUDIO_SHAPES[self.audio_features]


# The design ladder (EXPERIMENT-PLAN.md section 4): each step changes one thing.
PRESETS = {
    "m1": ModelSpec(),
    "d1": ModelSpec(visual="deep"),
    "d2": ModelSpec(visual="deep", gating=True),
    "d3": ModelSpec(visual="deep", gating=True, demographics=True),
    "d4": ModelSpec(visual="deep", gating=True, text=True),
    "m3": ModelSpec(visual="deep", gating=True, text=True, demographics=True),
    "audio_only": ModelSpec(video=False),
    "video_only": ModelSpec(audio=False),
}


@keras.saving.register_keras_serializable(package="mffer")
class Gate(layers.Layer):
    """Multiply an encoding by a learned scalar in (0, 1) computed from that encoding."""

    def build(self, input_shape):
        self.score = layers.Dense(1, activation="sigmoid")
        self.score.build(input_shape)

    def call(self, x):
        return x * self.score(x)


@keras.saving.register_keras_serializable(package="mffer")
class FrameMask(layers.Layer):
    """1 for a real audio frame, 0 for padding. The pipeline sets padded frames to exactly 0 after
    normalising (`pipeline._inputs`); a real frame is never all-zero. Optionally max-pooled by `pool` so it
    follows a time-downsampled encoder."""

    def __init__(self, pool: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.pool = pool

    def call(self, x):
        axes = tuple(range(2, len(x.shape)))
        mask = keras.ops.cast(keras.ops.any(keras.ops.not_equal(x, 0), axis=axes), "float32")
        if self.pool > 1:
            mask = keras.ops.max(keras.ops.reshape(mask, (-1, mask.shape[1] // self.pool, self.pool)), axis=-1)
        return mask

    def get_config(self):
        return {**super().get_config(), "pool": self.pool}


@keras.saving.register_keras_serializable(package="mffer")
class MaskedMean(layers.Layer):
    """Mean over time of the frames where `mask` is 1: (batch, time, channels), (batch, time) -> (batch, channels)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True   # accepts the mask BatchNormalization attaches; the output has no time axis

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, x, mask):
        weights = keras.ops.expand_dims(mask, -1)
        return keras.ops.sum(x * weights, axis=1) / keras.ops.maximum(keras.ops.sum(weights, axis=1), 1.0)


@keras.saving.register_keras_serializable(package="mffer")
class MelMean(layers.Layer):
    """Mean over the mel axis: (batch, time, mel, channels) -> (batch, time, channels)."""

    def call(self, x):
        return keras.ops.mean(x, axis=2)


def audio_encoder(x, spec: "ModelSpec"):
    masked = spec.audio_pool == "masked"
    if spec.audio_features == "logmel":
        return logmel_encoder(x, masked, units=64 * spec.width)
    mask = FrameMask()(x) if masked else None
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    if masked:   # padding excluded from the batch statistics and from the mean (ADR 0018)
        x = MaskedMean()(layers.BatchNormalization()(x, mask=mask), mask)
    else:
        x = layers.GlobalAveragePooling1D()(layers.BatchNormalization()(x))
    return layers.Dense(64 * spec.width, activation="relu")(x)


def logmel_encoder(x, masked: bool, units: int):
    """3 x [Conv2D 3x3 + BN + MaxPool 2] over (time, mel), mean over mel, then mean over time (ADR 0020)."""
    frames = x
    x = layers.Reshape((*x.shape[1:], 1))(x)
    for filters in (32, 64, 128):   # BN unmasked here; the time mean below is masked
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
    x = MelMean()(x)
    if masked:
        x = MaskedMean()(x, FrameMask(pool=8)(frames))   # 3 x MaxPool 2: time / 8
    else:
        x = layers.GlobalAveragePooling1D()(x)
    return layers.Dense(units, activation="relu")(x)


def visual_encoder(x, kind: str, blocks: int = 3, width: int = 1):
    td = layers.TimeDistributed
    if kind == "simple":
        x = td(layers.Conv2D(16, 3, strides=2, activation="relu"))(x)
        x = td(layers.MaxPooling2D(4))(x)
        x = td(layers.GlobalAveragePooling2D())(x)
        return layers.LSTM(32)(x)
    if kind == "deep":
        for filters in (32, 64, 128, 256)[:blocks]:
            x = td(layers.Conv2D(filters, 3, strides=2 if filters == 32 else 1,
                                 padding="same", activation="relu"))(x)
            x = td(layers.BatchNormalization())(x)
            x = td(layers.MaxPooling2D(2))(x)
        x = td(layers.GlobalAveragePooling2D())(x)
        x = layers.Bidirectional(layers.LSTM(64 * width))(x)
        return layers.Dense(128 * width, activation="relu")(x)
    raise ValueError(f"unknown visual encoder: {kind}")


def fusion_block(x, units: int, dropout: float):
    x = layers.Dense(units, activation="relu")(x)
    return layers.Dropout(dropout)(x)


def build_model(spec: ModelSpec, coverage: dict[str, float] | None = None) -> keras.Model:
    """`coverage`: share of training clips that carry each head's label. Keras averages a masked head's
    loss over the whole batch, so its weight is divided by its coverage to keep the configured weight."""
    assert spec.audio or spec.video, "a model needs audio or video"
    assert (spec.audio and spec.video) or not (spec.text or spec.demographics or spec.gating), \
        "text, demographics and gating are defined for the audio-visual model only"
    inputs, encodings = {}, []
    if spec.audio:
        inputs["audio"] = keras.Input(spec.audio_shape, name="audio")
        audio = audio_encoder(inputs["audio"], spec)
        encodings.append(audio)
    if spec.video:
        inputs["video"] = keras.Input(VIDEO_SHAPE, name="video")
        visual = visual_encoder(inputs["video"], spec.visual, spec.visual_blocks, spec.width)
        encodings.append(visual)
    if spec.gating:
        audio, visual = Gate(name="gate_audio")(audio), Gate(name="gate_visual")(visual)

    acoustic_group = [audio] if spec.audio else []   # groups only matter with text/demographics,
    visual_group = [visual] if spec.video else []     # which require both modalities (asserted above)
    if spec.text:
        inputs["text"] = keras.Input((TEXT_DIM,), name="text")
        text = layers.Dropout(0.2)(layers.Dense(64, activation="relu")(inputs["text"]))
        if spec.gating:
            text = Gate(name="gate_text")(text)
        acoustic_group.append(text)
    if spec.demographics:
        inputs["demographics"] = keras.Input((DEMO_DIM,), name="demographics")
        demo = layers.Dropout(0.2)(layers.Dense(32, activation="relu")(inputs["demographics"]))
        acoustic_group.append(demo)
        visual_group.append(demo)

    if spec.text or spec.demographics:   # hierarchical fusion: two groups, then merge
        fused = layers.Concatenate()([
            fusion_block(layers.Concatenate()(acoustic_group) if len(acoustic_group) > 1 else audio, 128 * spec.width, 0.3),
            fusion_block(layers.Concatenate()(visual_group) if len(visual_group) > 1 else visual, 128 * spec.width, 0.3),
        ])
    elif len(encodings) == 2:            # feature-level fusion: plain concatenation
        fused = layers.Concatenate()([audio, visual])
    else:                                # unimodal baseline
        fused = encodings[0]
    shared = layers.Dense(128 * spec.width, activation="relu")(fusion_block(fused, 256 * spec.width, 0.4))

    outputs = {name: layers.Dense(HEADS[name][0], activation=HEADS[name][1], name=name)(shared)
               for name in spec.heads}
    return compile_model(keras.Model(inputs, outputs, name="mffer"), spec, coverage)


def compile_model(model: keras.Model, spec: ModelSpec, coverage: dict[str, float] | None = None) -> keras.Model:
    """(Re)compile with a fresh Adam. Federated clients call this every round, with their own coverage."""
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss={name: HEADS[name][2] for name in spec.heads},
        loss_weights={h: w / (coverage or {}).get(h, 1.0) for h, w in spec.loss_weights.items()},
        # weighted, so masked clips (sample weight 0) don't count in the logged metrics either. Macro-F1 per
        # classification head: argmax per clip, accumulated over the epoch; equals metrics.macro_f1 (tested).
        weighted_metrics={name: ["accuracy", keras.metrics.F1Score(average="macro", name="macro_f1")]
                          if HEADS[name][2] != "mse" else ["mae"] for name in spec.heads},
    )
    return model


def derive_seed(*parts: int) -> int:
    """A 32-bit seed from integers (e.g. run seed, epoch): the one rule for every derived seed (ADR 0016)."""
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0])


def reseed(model: keras.Model, seed: int) -> None:
    """Set every random layer's (dropout's) state from `seed`. Each layer keeps its own seed generator that
    advances with every step and is rebuilt when a model is loaded, so without this the masks of an epoch
    (or round) would depend on what ran earlier in the process: a resumed run would differ from an
    uninterrupted one, and a federated client from a fresh worker process (ADR 0016)."""
    generators = [g for layer in model._flatten_layers(include_self=False)
                  for g in getattr(layer, "_seed_generators", [])]
    for i, generator in enumerate(generators):
        generator.state.assign(keras.ops.convert_to_tensor([derive_seed(seed, i), 0], generator.state.dtype))


def spec_from_config(model_config: dict) -> ModelSpec:
    """`model: {preset: m3, ...overrides}` in a run YAML."""
    config = dict(model_config)
    return replace(PRESETS[config.pop("preset")], **config)
