"""Input pipeline: rows of CREMA-D -> batches of (inputs, targets, sample_weights) for Keras.

Design and decisions: docs/pipeline-design.md. Owns three invariants:
- normalisation statistics come only from the rows the model trains on (`fit_stats`);
- a missing label (tied perceived vote, displayed intensity outside IEO) gets sample weight 0;
- the same rows, seed and statistics give the same batches; evaluation keeps row order.
"""
import json
from dataclasses import asdict, dataclass
from itertools import count
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

from mffer.config import load_paths
from mffer.data.cremad import EMOTIONS, VALENCES
from mffer.models import AUDIO_SHAPE, DEMO_DIM, HEADS, TEXT_DIM, VIDEO_SHAPE, ModelSpec

SENTENCES = ["DFA", "IEO", "IOM", "ITH", "ITS", "IWL", "IWW", "MTI", "TAI", "TIE", "TSI", "WSI"]
assert len(SENTENCES) == TEXT_DIM


class Cremad:
    """clips.csv and cremad.h5, opened once per process. Row i of one is row i of the other."""

    def __init__(self, processed: Path | None = None):
        processed = processed or load_paths().processed
        self.clips = pd.read_csv(processed / "clips.csv")
        self.h5 = h5py.File(processed / "cremad.h5", "r")
        stored = self.h5["clip_id"][:].astype(str)
        assert (stored == self.clips["clip_id"].to_numpy()).all(), "clips.csv and cremad.h5 rows differ"
        self.targets, self.weights = encode_labels(self.clips)

        self._audio = self.h5["audio"][:]                 # 3.9 MB, kept in memory
        self._audio_frames = self.h5["audio_frames"][:]

    def read(self, rows: np.ndarray, video: bool = True) -> dict[str, np.ndarray]:
        """Audio (+ video) for `rows`, in the given order. Video is read one clip (= one chunk) at a time:
        h5py's multi-row selection is ~80x slower (0.91 vs 0.011 s per batch of 16, measured 18 Nov)."""
        out = {"audio": self._audio[rows], "audio_frames": self._audio_frames[rows]}
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


def fit_stats(data: Cremad, rows: np.ndarray) -> AudioStats:
    audio = data.read(rows, video=False)
    valid = np.arange(AUDIO_SHAPE[0])[None, :] < audio["audio_frames"][:, None]
    frames = audio["audio"][valid].astype(np.float64)   # float32 sums over ~300k frames drift
    return AudioStats(frames.mean(0).tolist(), frames.std(0).tolist(), int(len(frames)))


def _inputs(data: Cremad, rows: np.ndarray, spec: ModelSpec, stats: AudioStats) -> dict[str, np.ndarray]:
    raw = data.read(rows, video=spec.video)
    x = {}
    if spec.audio:
        audio = (raw["audio"] - np.float32(stats.mean)) / np.float32(stats.std)
        audio *= (np.arange(AUDIO_SHAPE[0])[None, :] < raw["audio_frames"][:, None])[..., None]
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


def make_dataset(data: Cremad, rows: np.ndarray, spec: ModelSpec, stats: AudioStats, *,
                 training: bool, seed: int = 0, batch_size: int = 16) -> tf.data.Dataset:
    """Batches of (inputs, targets, sample_weights). Training: reshuffled each epoch with seed + epoch.
    Evaluation: in `rows` order, last batch may be smaller."""
    rows = np.asarray(rows)
    epochs = count()

    def batches():
        order = np.random.default_rng(seed + next(epochs)).permutation(rows) if training else rows
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            yield (_inputs(data, batch, spec, stats),
                   {h: data.targets[h][batch] for h in spec.heads},
                   {h: data.weights[h][batch] for h in spec.heads})

    x_sig = {}
    if spec.audio:
        x_sig["audio"] = tf.TensorSpec((None, *AUDIO_SHAPE), tf.float32)
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
