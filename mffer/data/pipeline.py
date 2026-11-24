"""Input pipeline: rows of CREMA-D -> batches of (inputs, targets, sample_weights) for Keras.

Design and decisions: docs/pipeline-design.md. Owns three invariants:
- normalisation statistics come only from the rows the model trains on (`fit_stats`);
- a missing label (tied perceived vote, displayed intensity outside IEO) gets sample weight 0;
- the same rows, seed and statistics give the same batches; evaluation keeps row order.

Protocol-2 options, all read from the `ModelSpec` and applied to training datasets only (never to
validation or test): valence class weights (`balance_valence`, ADR 0019) and augmentation (`augment`,
ADR 0021). Audio is MFCC (`cremad.h5`) or log-mel (`logmel.h5`, ADR 0020), per `spec.audio_features`.
"""
import json
from dataclasses import asdict, dataclass
from itertools import count
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

from mffer.config import load_paths
from mffer.data.cremad import EMOTIONS, VALENCES
from mffer.models import AUDIO_SHAPES, DEMO_DIM, HEADS, TEXT_DIM, VIDEO_SHAPE, ModelSpec, derive_seed

SENTENCES = ["DFA", "IEO", "IOM", "ITH", "ITS", "IWL", "IWW", "MTI", "TAI", "TIE", "TSI", "WSI"]
assert len(SENTENCES) == TEXT_DIM


AUDIO_FILES = {"mfcc": "cremad.h5", "logmel": "logmel.h5"}


class Cremad:
    """clips.csv and cremad.h5 (+ logmel.h5 when a model uses it), opened once per process. Row i of one is
    row i of the others."""

    def __init__(self, processed: Path | None = None):
        self.processed = processed or load_paths().processed
        self.clips = pd.read_csv(self.processed / "clips.csv")
        self.h5 = self._open("cremad.h5")
        self.targets, self.weights = encode_labels(self.clips)
        self._audio = {"mfcc": (self.h5["audio"][:], self.h5["audio_frames"][:])}   # 3.9 MB, kept in memory

    def _open(self, name: str) -> h5py.File:
        h5 = h5py.File(self.processed / name, "r")
        stored = h5["clip_id"][:].astype(str)
        assert (stored == self.clips["clip_id"].to_numpy()).all(), f"clips.csv and {name} rows differ"
        return h5

    def audio(self, features: str) -> tuple[np.ndarray, np.ndarray]:
        """(features, valid frame counts) for every clip; log-mel (0.8 GB) is loaded on first use."""
        if features not in self._audio:
            h5 = self._open(AUDIO_FILES[features])
            self._audio[features] = (h5["audio"][:], h5["audio_frames"][:])
        return self._audio[features]

    def read(self, rows: np.ndarray, video: bool = True, features: str = "mfcc") -> dict[str, np.ndarray]:
        """Audio (+ video) for `rows`, in the given order. Video is read one clip (= one chunk) at a time:
        h5py's multi-row selection is ~80x slower (0.91 vs 0.011 s per batch of 16, measured 18 Nov)."""
        audio, frames = self.audio(features)
        out = {"audio": audio[rows], "audio_frames": frames[rows]}
        if video:
            out["video"] = np.stack([self.h5["video"][i] for i in rows])
        return out


def encode_labels(clips: pd.DataFrame) -> tuple[dict, dict]:
    """Targets and 0/1 sample weights for every head, for all clips (masked targets are zeros)."""
    def one_hot(column, classes):
        index = clips[column].map({c: i for i, c in enumerate(classes)})
        present = index.notna().to_numpy()
        out = np.zeros((len(clips), len(classes)), np.float32)
        out[np.flatnonzero(present), index[present].astype(int)] = 1
        return out, present.astype(np.float32)

    def scalar(column):
        values = clips[column].to_numpy(np.float32)
        present = ~np.isnan(values)
        return np.where(present, values, 0).reshape(-1, 1).astype(np.float32), present.astype(np.float32)

    encoded = {"displayed_emotion": one_hot("displayed_emotion", EMOTIONS),
               "perceived_emotion": one_hot("perceived_emotion", EMOTIONS),
               "displayed_valence": one_hot("displayed_valence", VALENCES),
               "displayed_intensity": scalar("displayed_intensity"),
               "perceived_intensity": scalar("perceived_intensity")}
    return {k: v[0] for k, v in encoded.items()}, {k: v[1] for k, v in encoded.items()}


def label_coverage(data: Cremad, rows: np.ndarray, spec: ModelSpec) -> dict[str, float]:
    """Share of `rows` that carry a label, per head. Used to compensate the loss weights of masked heads."""
    return {head: float(data.weights[head][rows].mean()) for head in spec.heads}


@dataclass
class AudioStats:
    """Per-MFCC-coefficient mean and std over valid (non-padded) frames."""
    mean: list[float]
    std: list[float]
    n_frames: int

    def save(self, file: Path) -> None:
        file.write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, file: Path) -> "AudioStats":
        return cls(**json.loads(file.read_text()))

    @classmethod
    def combine(cls, parts: list["AudioStats"]) -> "AudioStats":
        """Exact pooled statistics from per-client statistics (federated runs share only these numbers)."""
        n = np.array([p.n_frames for p in parts], float)
        means, stds = np.array([p.mean for p in parts]), np.array([p.std for p in parts])
        mean = (n[:, None] * means).sum(0) / n.sum()
        var = (n[:, None] * (stds ** 2 + (means - mean) ** 2)).sum(0) / n.sum()
        return cls(mean.tolist(), np.sqrt(var).tolist(), int(n.sum()))


def valence_class_weights(data: Cremad, rows: np.ndarray) -> list[float]:
    """Inverse-frequency weight per valence class (VALENCES order) on `rows`: n / (classes x n_class), so the
    mean weight over `rows` is exactly 1 and the head's loss scale (and ADR 0006's coverage) is unchanged."""
    counts = data.targets["displayed_valence"][rows].sum(0)
    return (counts.sum() / (len(counts) * counts)).tolist()


def fit_stats(data: Cremad, rows: np.ndarray, features: str = "mfcc") -> AudioStats:
    audio = data.read(rows, video=False, features=features)
    valid = np.arange(AUDIO_SHAPES[features][0])[None, :] < audio["audio_frames"][:, None]
    frames = audio["audio"][valid].astype(np.float64)   # float32 sums over ~300k frames drift
    return AudioStats(frames.mean(0).tolist(), frames.std(0).tolist(), int(len(frames)))


def _inputs(data: Cremad, rows: np.ndarray, spec: ModelSpec, stats: AudioStats) -> dict[str, np.ndarray]:
    raw = data.read(rows, video=spec.video, features=spec.audio_features)
    x = {}
    if spec.audio:
        audio = (raw["audio"] - np.float32(stats.mean)) / np.float32(stats.std)
        audio *= (np.arange(spec.audio_shape[0])[None, :] < raw["audio_frames"][:, None])[..., None]
        x["audio"] = audio.astype(np.float32)
    if spec.video:
        x["video"] = raw["video"]                                             # uint8 until the map
    clips = data.clips.iloc[rows]
    if spec.text:
        x["text"] = np.eye(TEXT_DIM, dtype=np.float32)[clips["sentence"].map(SENTENCES.index).to_numpy()]
    if spec.demographics:
        x["demographics"] = np.stack([clips["age"].to_numpy() / 100,
                                      (clips["sex"] == "Male").to_numpy()], axis=1).astype(np.float32)
    return x


def augment(x: dict[str, np.ndarray], spec: ModelSpec, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Training-only augmentation of one batch (ADR 0021), drawn from `rng` in a fixed order so a seed
    reproduces it. Video: per clip, the same horizontal flip (p 0.5) and random square crop of 200-224 px,
    resized back to 224, for all 10 frames. Log-mel: per clip, 2 time masks (each up to 10% of the real
    frames) and 2 frequency masks (each up to 8 bands), set to 0 (the normalised mean). MFCC audio is left
    as is (masking cepstral coefficients has no spectral meaning)."""
    x = dict(x)
    if "video" in x:
        video = x["video"].copy()
        size = VIDEO_SHAPE[1]
        for clip in range(len(video)):
            flip, side = rng.random() < 0.5, int(rng.integers(200, size + 1))
            top, left = (int(v) for v in rng.integers(0, size - side + 1, 2))
            for f in range(len(video[clip])):
                frame = video[clip, f, top:top + side, left:left + side]
                frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA) if side < size else frame
                video[clip, f] = frame[:, ::-1] if flip else frame
        x["video"] = video
    if "audio" in x and spec.audio_features == "logmel":
        audio = x["audio"].copy()
        n_bands = audio.shape[2]
        for clip in range(len(audio)):
            real = int(np.any(audio[clip] != 0, axis=1).sum())
            for _ in range(2):
                width = int(rng.integers(0, max(real // 10, 1) + 1))
                start = int(rng.integers(0, max(real - width, 0) + 1))
                audio[clip, start:start + width] = 0
            for _ in range(2):
                width = int(rng.integers(0, 9))
                start = int(rng.integers(0, n_bands - width + 1))
                audio[clip, :real, start:start + width] = 0
        x["audio"] = audio
    return x


def make_dataset(data: Cremad, rows: np.ndarray, spec: ModelSpec, stats: AudioStats, *,
                 training: bool, seed: int = 0, batch_size: int = 16, first_epoch: int = 0) -> tf.data.Dataset:
    """Batches of (inputs, targets, sample_weights). Training: reshuffled each epoch with
    derive_seed(seed, epoch), counting from `first_epoch` (a resumed run passes the epochs already done, so
    it shuffles as an uninterrupted run would); valence class weights from these rows and augmentation
    from derive_seed(seed, epoch, 1) when the spec asks. Evaluation: in `rows` order, 0/1 weights, no
    augmentation; last batch may be smaller."""
    rows = np.asarray(rows)
    epochs = count(first_epoch)
    valence = np.float32(valence_class_weights(data, rows)) if training and spec.balance_valence else None

    def batches():
        epoch = next(epochs)
        order = np.random.default_rng(derive_seed(seed, epoch)).permutation(rows) if training else rows
        rng = np.random.default_rng(derive_seed(seed, epoch, 1))
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            x = _inputs(data, batch, spec, stats)
            if training and spec.augment:
                x = augment(x, spec, rng)
            w = {h: data.weights[h][batch] for h in spec.heads}
            if valence is not None:
                classes = data.targets["displayed_valence"][batch].argmax(1)
                w["displayed_valence"] = w["displayed_valence"] * valence[classes]
            yield x, {h: data.targets[h][batch] for h in spec.heads}, w

    x_sig = {}
    if spec.audio:
        x_sig["audio"] = tf.TensorSpec((None, *spec.audio_shape), tf.float32)
    if spec.video:
        x_sig["video"] = tf.TensorSpec((None, *VIDEO_SHAPE), tf.uint8)
    if spec.text:
        x_sig["text"] = tf.TensorSpec((None, TEXT_DIM), tf.float32)
    if spec.demographics:
        x_sig["demographics"] = tf.TensorSpec((None, DEMO_DIM), tf.float32)
    y_sig = {h: tf.TensorSpec((None, HEADS[h][0]), tf.float32) for h in spec.heads}
    w_sig = {h: tf.TensorSpec((None,), tf.float32) for h in spec.heads}

    def to_float(x, y, w):
        return {**x, "video": tf.cast(x["video"], tf.float32) / 255.0}, y, w

    dataset = tf.data.Dataset.from_generator(batches, output_signature=(x_sig, y_sig, w_sig))
    # Declared length: Keras then knows the epoch size and doesn't warn that the input "ran out of data".
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(-(-len(rows) // batch_size)))
    if spec.video:
        dataset = dataset.map(to_float, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(2)
