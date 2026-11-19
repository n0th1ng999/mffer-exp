"""One model builder for every configuration (M1, the design-ladder steps, M3).

Configurations differ only by the flags in `ModelSpec`, so a step of the ladder changes exactly what its
flags say. Layer sizes follow Table 1 of the paper draft.
"""
from dataclasses import dataclass, field, replace

import keras
from keras import layers

N_EMOTIONS = 6
N_VALENCE = 3
AUDIO_SHAPE = (100, 13)            # MFCC frames x coefficients
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
    loss_weights: dict = field(default_factory=lambda: {
        "displayed_emotion": 1.0, "perceived_emotion": 1.0, "displayed_valence": 1.0,
        "displayed_intensity": 0.2, "perceived_intensity": 0.2,
    })

    @property
    def heads(self) -> list[str]:
        return list(self.loss_weights)


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


def audio_encoder(x):
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    return layers.Dense(64, activation="relu")(x)


def visual_encoder(x, kind: str):
    td = layers.TimeDistributed
    if kind == "simple":
        x = td(layers.Conv2D(16, 3, strides=2, activation="relu"))(x)
        x = td(layers.MaxPooling2D(4))(x)
        x = td(layers.GlobalAveragePooling2D())(x)
        return layers.LSTM(32)(x)
    if kind == "deep":
        for filters in (32, 64, 128):
            x = td(layers.Conv2D(filters, 3, strides=2 if filters == 32 else 1,
                                 padding="same", activation="relu"))(x)
            x = td(layers.BatchNormalization())(x)
            x = td(layers.MaxPooling2D(2))(x)
        x = td(layers.GlobalAveragePooling2D())(x)
        x = layers.Bidirectional(layers.LSTM(64))(x)
        return layers.Dense(128, activation="relu")(x)
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
        inputs["audio"] = keras.Input(AUDIO_SHAPE, name="audio")
        audio = audio_encoder(inputs["audio"])
        encodings.append(audio)
    if spec.video:
        inputs["video"] = keras.Input(VIDEO_SHAPE, name="video")
        visual = visual_encoder(inputs["video"], spec.visual)
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
            fusion_block(layers.Concatenate()(acoustic_group) if len(acoustic_group) > 1 else audio, 128, 0.3),
            fusion_block(layers.Concatenate()(visual_group) if len(visual_group) > 1 else visual, 128, 0.3),
        ])
    elif len(encodings) == 2:            # feature-level fusion: plain concatenation
        fused = layers.Concatenate()([audio, visual])
    else:                                # unimodal baseline
        fused = encodings[0]
    shared = layers.Dense(128, activation="relu")(fusion_block(fused, 256, 0.4))

    outputs = {name: layers.Dense(HEADS[name][0], activation=HEADS[name][1], name=name)(shared)
               for name in spec.heads}
    return compile_model(keras.Model(inputs, outputs, name="mffer"), spec, coverage)


def compile_model(model: keras.Model, spec: ModelSpec, coverage: dict[str, float] | None = None) -> keras.Model:
    """(Re)compile with a fresh Adam. Federated clients call this every round, with their own coverage."""
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss={name: HEADS[name][2] for name in spec.heads},
        loss_weights={h: w / (coverage or {}).get(h, 1.0) for h, w in spec.loss_weights.items()},
        # weighted, so masked clips (sample weight 0) don't count in the logged metrics either
        weighted_metrics={name: ["accuracy"] if HEADS[name][2] != "mse" else ["mae"] for name in spec.heads},
    )
    return model


def spec_from_config(model_config: dict) -> ModelSpec:
    """`model: {preset: m3, ...overrides}` in a run YAML."""
    config = dict(model_config)
    return replace(PRESETS[config.pop("preset")], **config)
