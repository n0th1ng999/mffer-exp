"""RAVDESS probes for the EDA (notebooks/eda_ravdess.ipynb, sections 9-10): zero-shot transfer of trained
CREMA-D models to RAVDESS, and Grad-CAM saliency of their visual encoders. Exploration only: nothing here is
used by training, and CREMA-D's test actors are never touched (RAVDESS is the test set; saliency on CREMA-D uses
split 0's validation actors).

    MFFER_CPU=1 python scripts/ravdess_probe.py features            # RAVDESS input variants -> ../data/RAVDESS/probe/
    MFFER_CPU=1 python scripts/ravdess_probe.py zero-shot runs/c1_m1-s0-... [more run dirs]
    MFFER_CPU=1 python scripts/ravdess_probe.py saliency runs/c1_m1-s0-...

Input variants (cumulative ladder, one target-side change per step; the model is never retrained):
  A  as is: whole-file MFCC with the run's CREMA-D audio statistics, 10 frames over the whole file, centre crop
  B  + audio silence trimmed (librosa top_db 30)
  C  + RAVDESS's own audio statistics (per-client normalisation)
  D  + the 10 frames sampled inside the speech interval
  E  + CREMA-D-like framing: zoom out (edge-replicated border) so the face has CREMA-D's median size and position
Labels: the 6 shared emotions (calm, surprised dropped); intensity normal -> 2/3, strong -> 1, neutral masked.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from mffer.config import load_paths
from mffer.data.cremad import EMOTIONS, VALENCE, VALENCES
from mffer.data.features import HOP, N_FFT, N_MFCC, SAMPLE_RATE, SIZE
from mffer.models import AUDIO_SHAPE, VIDEO_SHAPE

PATHS = load_paths()
RAV = PATHS.raw.parent / "RAVDESS"
OUT = RAV / "probe"
YUNET = PATHS.raw.parent / "models" / "face_detection_yunet_2023mar.onnx"
EMOTION = {1: "NEU", 3: "HAP", 4: "SAD", 5: "ANG", 6: "FEA", 7: "DIS"}        # calm (2), surprised (8) dropped
CREMAD_FACE = {"width": 0.35, "cx": 0.49, "cy": 0.44}                          # notebook §8c, median per clip
TOP_DB = 30
N_FRAMES = VIDEO_SHAPE[0]


# ---------------------------------------------------------------- clip table

def clip_table() -> pd.DataFrame:
    rows = []
    for path in sorted(RAV.glob("video_speech/**/01-01-*.mp4")):
        f = list(map(int, re.match(r"^(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)\.mp4$", path.name).groups()))
        if f[2] not in EMOTION:
            continue
        emotion = EMOTION[f[2]]
        rows.append(dict(video=str(path), wav=str(RAV / "audio_speech" / f"Actor_{f[6]:02d}" / ("03" + path.name[2:-4] + ".wav")),
                         actor=f[6], sex="Male" if f[6] % 2 else "Female", emotion=emotion, valence=VALENCE[emotion],
                         intensity={1: "normal", 2: "strong"}[f[3]], statement=f[4], repetition=f[5]))
    table = pd.DataFrame(rows)
    table["displayed_intensity"] = np.where(table.emotion == "NEU", np.nan,
                                            np.where(table.intensity == "normal", 2 / 3, 1.0))
    return table


# ---------------------------------------------------------------- audio

def mfcc(wav: str, trim: bool) -> tuple[np.ndarray, int, float, float, float]:
    """Same MFCC as mffer.data.features.mfcc (16 kHz, 13 coefficients, n_fft 1024, hop 640, 100 frames),
    optionally on the silence-trimmed signal. Also returns the speech interval (s) and the file length."""
    signal, _ = librosa.load(wav, sr=SAMPLE_RATE)
    _, (start, end) = librosa.effects.trim(signal, top_db=TOP_DB)
    used = signal[start:end] if trim else signal
    coeffs = librosa.feature.mfcc(y=used, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP).T
    n = min(len(coeffs), AUDIO_SHAPE[0])
    out = np.zeros(AUDIO_SHAPE, np.float32)
    out[:n] = coeffs[:n]
    return out, n, start / SAMPLE_RATE, end / SAMPLE_RATE, len(signal) / SAMPLE_RATE


# ---------------------------------------------------------------- video

def _detector(size):
    return cv2.FaceDetectorYN.create(str(YUNET), "", size, 0.6)


def _square(frame: np.ndarray) -> np.ndarray:
    """mffer.data.features.frames: centre square crop, resized to 224, RGB."""
    h, w = frame.shape[:2]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    return cv2.cvtColor(cv2.resize(frame[top:top + side, left:left + side], (SIZE, SIZE), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2RGB)


def _cremad_framing(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Crop a square around the face so the face has CREMA-D's median width share and centre position;
    the image is edge-replicated where the square leaves the frame (RAVDESS's background is plain white)."""
    x, y, w, h = box
    side = w / CREMAD_FACE["width"]
    left = x + w / 2 - CREMAD_FACE["cx"] * side
    top = y + h / 2 - CREMAD_FACE["cy"] * side
    pad = int(side)
    padded = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    x0, y0, s = int(round(left)) + pad, int(round(top)) + pad, int(round(side))
    crop = padded[y0:y0 + s, x0:x0 + s]
    return cv2.cvtColor(cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def video_variants(path: str, speech: tuple[float, float]) -> dict[str, np.ndarray]:
    """10 frames three ways: whole file + centre crop (A-C), speech interval + centre crop (D),
    speech interval + CREMA-D-like framing (E)."""
    capture = cv2.VideoCapture(path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    decoded = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded.append(frame)
    capture.release()
    n = len(decoded)
    whole = np.linspace(0, n - 1, N_FRAMES).round().astype(int)
    first, last = int(np.clip(speech[0] * fps, 0, n - 1)), int(np.clip(speech[1] * fps, 0, n - 1))
    inside = np.linspace(first, max(first, last), N_FRAMES).round().astype(int)
    h, w = decoded[0].shape[:2]
    detector = _detector((w // 2, h // 2))
    boxes = []
    for i in inside:
        _, found = detector.detect(cv2.resize(decoded[i], (w // 2, h // 2)))
        if found is not None:
            boxes.append(2 * max(found, key=lambda f: f[2] * f[3])[:4])
    box = np.median(np.array(boxes), axis=0) if boxes else np.array([w / 2 - h / 4, h / 4, h / 2, h / 2])
    return {"whole": np.stack([_square(decoded[i]) for i in whole]),
            "speech": np.stack([_square(decoded[i]) for i in inside]),
            "framed": np.stack([_cremad_framing(decoded[i], box) for i in inside]),
            "face_found": len(boxes) / N_FRAMES}


def build_features() -> None:
    OUT.mkdir(exist_ok=True)
    table = clip_table()
    print(f"{len(table)} clips, {table.actor.nunique()} actors", flush=True)

    def one(row):
        raw, n_raw, start, end, length = mfcc(row.wav, trim=False)
        trimmed, n_trim, *_ = mfcc(row.wav, trim=True)
        video = video_variants(row.video, (start, end))
        return raw, n_raw, trimmed, n_trim, video, dict(speech_start=start, speech_end=end, length=length,
                                                        face_found=video["face_found"])

    n = len(table)
    arrays = {k: np.lib.format.open_memmap(OUT / f"video_{k}.npy", "w+", np.uint8, (n, *VIDEO_SHAPE))
              for k in ("whole", "speech", "framed")}
    audio = {k: np.zeros((n, *AUDIO_SHAPE), np.float32) for k in ("raw", "trimmed")}
    frames = {k: np.zeros(n, np.int16) for k in ("raw", "trimmed")}
    extra = []
    for chunk in range(0, n, 32):                            # bounded memory: 32 decoded clips at a time
        results = Parallel(n_jobs=4)(delayed(one)(row) for row in table.iloc[chunk:chunk + 32].itertuples())
        for i, (raw, n_raw, trimmed, n_trim, video, info) in enumerate(results, start=chunk):
            audio["raw"][i], frames["raw"][i], audio["trimmed"][i], frames["trimmed"][i] = raw, n_raw, trimmed, n_trim
            for k in arrays:
                arrays[k][i] = video[k]
            extra.append(info)
        print(f"  {min(chunk + 32, n)}/{n}", flush=True)
    for k in arrays:
        arrays[k].flush()
    np.savez(OUT / "audio.npz", **{f"audio_{k}": v for k, v in audio.items()}, **{f"frames_{k}": v for k, v in frames.items()})
    pd.concat([table, pd.DataFrame(extra)], axis=1).to_csv(OUT / "clips.csv", index=False)
    print("features written to", OUT)


# ---------------------------------------------------------------- models

def load_run(run_dir: Path):
    import yaml

    from mffer.data.pipeline import AudioStats
    from mffer.models import build_model, spec_from_config
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    spec = spec_from_config(config["model"])
    if spec.text or spec.demographics:
        sys.exit(f"{run_dir.name}: uses text/demographics, which RAVDESS cannot fill (EDA §7b); use D2, C1 or U1")
    model = build_model(spec)
    model.load_weights(run_dir / "best.weights.h5")
    return config, spec, model, AudioStats.load(run_dir / "audio_stats.json")


def normalise(audio, n_frames, mean, std):
    x = (audio - np.float32(mean)) / np.float32(std)
    return (x * (np.arange(AUDIO_SHAPE[0])[None, :] < n_frames[:, None])[..., None]).astype(np.float32)


def own_stats(audio, n_frames):
    valid = np.arange(AUDIO_SHAPE[0])[None, :] < n_frames[:, None]
    frames = audio[valid]
    return frames.mean(0), frames.std(0)


VARIANTS = {"A": ("raw", "run", "whole"), "B": ("trimmed", "run", "whole"), "C": ("trimmed", "own", "whole"),
            "D": ("trimmed", "own", "speech"), "E": ("trimmed", "own", "framed")}


def predict(model, spec, x_audio, video, batch=32):
    import tensorflow as tf
    outputs = {}
    for start in range(0, len(x_audio), batch):
        x = {}
        if spec.audio:
            x["audio"] = tf.constant(x_audio[start:start + batch])
        if spec.video:
            x["video"] = tf.cast(np.asarray(video[start:start + batch]), tf.float32) / 255.0
        for head, out in model(x, training=False).items():
            outputs.setdefault(head, []).append(np.asarray(out))
    return {h: np.concatenate(v) for h, v in outputs.items()}


def scores(table, out) -> dict:
    from sklearn.metrics import f1_score
    valence = np.array(VALENCES)[out["displayed_valence"].argmax(1)]
    emotion = np.array(EMOTIONS)[out["displayed_emotion"].argmax(1)]
    m = table.displayed_intensity.notna().to_numpy()
    result = {"valence_macro_f1": f1_score(table.valence, valence, average="macro", labels=VALENCES),
              "emotion_macro_f1": f1_score(table.emotion, emotion, average="macro", labels=EMOTIONS),
              "intensity_mae": float(np.abs(out["displayed_intensity"][m, 0] - table.displayed_intensity[m]).mean()),
              "predicted_negative_share": float((valence == "negative").mean())}
    for sex in ("Female", "Male"):
        s = (table.sex == sex).to_numpy()
        result[f"valence_macro_f1_{sex.lower()}"] = f1_score(table.valence[s], valence[s], average="macro", labels=VALENCES)
    return result


def zero_shot(run_dirs: list[Path]) -> None:
    table = pd.read_csv(OUT / "clips.csv")
    data = np.load(OUT / "audio.npz")
    videos = {k: np.load(OUT / f"video_{k}.npy", mmap_mode="r") for k in ("whole", "speech", "framed")}
    rows = []
    for run_dir in run_dirs:
        config, spec, model, stats = load_run(run_dir)
        in_domain = json.loads((run_dir / "test.json").read_text())["test"]   # already scored by the run itself
        for variant, (audio_kind, stats_kind, video_kind) in VARIANTS.items():
            if not spec.video and variant in ("D", "E"):
                continue                                     # audio-only model: video variants change nothing
            audio, n = data[f"audio_{audio_kind}"], data[f"frames_{audio_kind}"]
            mean, std = (stats.mean, stats.std) if stats_kind == "run" else own_stats(audio, n)
            out = predict(model, spec, normalise(audio, n, mean, std), videos[video_kind])
            rows.append({"run": run_dir.name, "config": config["name"], "variant": variant, **scores(table, out),
                         "cremad_test_valence_macro_f1": in_domain["displayed_valence"]["macro_f1"],
                         "cremad_test_emotion_macro_f1": in_domain["displayed_emotion"]["macro_f1"]})
            print(f"{run_dir.name} {variant}: valence F1 {rows[-1]['valence_macro_f1']:.3f} "
                  f"(CREMA-D test {rows[-1]['cremad_test_valence_macro_f1']:.3f})", flush=True)
    file = OUT / "zero_shot.csv"
    old = pd.read_csv(file) if file.exists() else pd.DataFrame(columns=["run", "variant"])
    old = old[~old.run.isin([r.name for r in run_dirs])]
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(file, index=False)
    print("written", file)


# ---------------------------------------------------------------- Grad-CAM

def grad_cam(model, x, head="displayed_valence"):
    """Grad-CAM (Selvaraju et al., ICCV 2017) on the last 5-D activation of the visual encoder
    (batch, time, height, width, channels), for the predicted class of `head`. Returns (batch, time, 224, 224)."""
    import keras
    import tensorflow as tf
    target = [layer for layer in model.layers if len(layer.output.shape) == 5][-1]
    probe = keras.Model(model.inputs, [target.output, model.output[head]])
    with tf.GradientTape() as tape:
        activation, prediction = probe(x, training=False)
        score = tf.reduce_sum(prediction * tf.one_hot(tf.argmax(prediction, 1), prediction.shape[1]), axis=1)
    grads = tape.gradient(score, activation)
    weights = tf.reduce_mean(grads, axis=(2, 3), keepdims=True)                 # per frame and channel
    cam = tf.nn.relu(tf.reduce_sum(weights * activation, axis=-1)).numpy()    # (b, t, h, w)
    b, t = cam.shape[:2]
    up = np.stack([cv2.resize(c, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR) for c in cam.reshape(b * t, *cam.shape[2:])])
    return up.reshape(b, t, SIZE, SIZE), target.name


def face_masks(frames: np.ndarray) -> np.ndarray:
    """(n, t, 224, 224, 3) -> boolean face masks from YuNet (box per frame; empty if not found)."""
    detector = _detector((SIZE, SIZE))
    masks = np.zeros(frames.shape[:4], bool)
    for i in range(frames.shape[0]):
        for t in range(frames.shape[1]):
            _, found = detector.detect(cv2.cvtColor(frames[i, t], cv2.COLOR_RGB2BGR))
            if found is not None:
                x, y, w, h = (int(round(v)) for v in max(found, key=lambda f: f[2] * f[3])[:4])
                masks[i, t, max(y, 0):y + h, max(x, 0):x + w] = True
    return masks


def saliency(run_dir: Path, n_clips: int = 96) -> None:
    import h5py

    from mffer.data.pipeline import Cremad
    from mffer.data.splits import rows as split_rows
    config, spec, model, stats = load_run(run_dir)
    if not spec.video:
        sys.exit(f"{run_dir.name}: no video input")
    rng = np.random.default_rng(0)
    cd = Cremad()
    split = json.loads((run_dir / "meta.json").read_text())["split"]
    val = split_rows(cd.clips, split, "val")                              # validation actors: test stays unseen
    pick_cd = np.sort(rng.choice(val, n_clips, replace=False))
    with h5py.File(PATHS.processed / "cremad.h5") as h5:
        frames_cd = np.stack([h5["video"][r] for r in pick_cd])
    audio_cd = cd.read(pick_cd, video=False)
    x_cd = normalise(audio_cd["audio"], audio_cd["audio_frames"], stats.mean, stats.std)

    table = pd.read_csv(OUT / "clips.csv")
    data = np.load(OUT / "audio.npz")
    pick_rv = np.sort(rng.choice(len(table), n_clips, replace=False))
    a_rv, n_rv = data["audio_trimmed"][pick_rv], data["frames_trimmed"][pick_rv]
    mean, std = own_stats(data["audio_trimmed"], data["frames_trimmed"])
    x_rv = normalise(a_rv, n_rv, mean, std)

    results, examples = [], {}
    for corpus, kind, frames, x_audio in [("CREMA-D", "as trained", frames_cd, x_cd)] + [
            ("RAVDESS", kind, np.asarray(np.load(OUT / f"video_{kind}.npy", mmap_mode="r")[pick_rv]), x_rv)
            for kind in ("whole", "speech", "framed")]:
        cams = []
        for s in range(0, n_clips, 16):
            x = {"video": frames[s:s + 16].astype(np.float32) / 255.0}
            if spec.audio:
                x["audio"] = x_audio[s:s + 16]
            cam, layer = grad_cam(model, x)
            cams.append(cam)
        cam = np.concatenate(cams)
        masks = face_masks(frames)
        total = cam.sum(axis=(2, 3)) + 1e-12
        in_face = (cam * masks).sum(axis=(2, 3)) / total                        # share of saliency on the face
        area = masks.mean(axis=(2, 3))
        ok = area > 0
        results.append({"run": run_dir.name, "corpus": corpus, "frames": kind, "layer": layer,
                        "saliency_on_face": float(np.median(in_face[ok])), "face_area": float(np.median(area[ok])),
                        "face_focus_ratio": float(np.median(in_face[ok] / area[ok])),
                        "frames_with_face": float(ok.mean())})
        examples[f"{corpus}|{kind}|frames"] = frames[:6, 5]
        examples[f"{corpus}|{kind}|cam"] = cam[:6, 5]
        print(results[-1], flush=True)
    pd.DataFrame(results).to_csv(OUT / f"saliency_{run_dir.name}.csv", index=False)
    np.savez_compressed(OUT / f"saliency_{run_dir.name}.npz", **examples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["features", "zero-shot", "saliency"])
    parser.add_argument("runs", nargs="*", type=Path)
    args = parser.parse_args()
    if args.command == "features":
        build_features()
    elif args.command == "zero-shot":
        zero_shot(args.runs)
    else:
        for run_dir in args.runs:
            saliency(run_dir)


if __name__ == "__main__":
    main()
