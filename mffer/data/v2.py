"""The V2 dataset (T0 data contract, ADR 0024): CREMA-D + RAVDESS (6 shared emotions) in one clip index, frozen
actor splits, and model A's features. Everything lives in `<processed>/v2/`; splits in `splits/v2/`.

    python -m mffer.data.v2 index        # clips.csv: both corpora, CREMA-D label schema
    python -m mffer.data.v2 splits       # splits/v2/split_<i>.json (frozen, never overwritten)
    python -m mffer.data.v2 features     # features.h5 + media.csv (trimmed-audio MFCC, aligned faces)

Labels (same columns as mffer.data.cremad, so the pipeline and metrics are unchanged):
- RAVDESS emotions neutral, happy, sad, angry, fearful, disgust -> NEU, HAP, SAD, ANG, FEA, DIS; calm and surprised
  are not used (EDA §4f). Valence by the same mapping as CREMA-D.
- displayed_intensity: RAVDESS normal -> 2/3, strong -> 1 (EDA §5c); neutral has no intensity (NaN, masked).
  displayed_level is "RN"/"RS" (not LO/MD/HI, which mean CREMA-D's IEO levels in the metrics).
- perceived_emotion / perceived_intensity / age: not available for RAVDESS -> NaN, masked per head (ADR 0006).
- actor_id: CREMA-D keeps 1001-1091; RAVDESS actor k becomes 2000 + k. `corpus` is "cremad" or "ravdess".

Splits: split i keeps V1 split i's CREMA-D actors exactly and adds RAVDESS's 24 actors with the same rule
(sex-stratified, 70/10/20, seed i): 18 train, 2 val, 4 test actors.

Features (row order = clips.csv, `clip_id` stored to check it): inside the speech interval (mffer.data.media),
13 MFCCs (as V1: n_fft 1024, hop 640, 100 frames, zero-padded) and 10 aligned, lighting-normalised face frames
(10 x 224 x 224 x 3, RGB uint8). media.csv records each clip's speech interval and face-detection share.
"""
import argparse
import json
import re
from multiprocessing import Pool

import h5py
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from mffer.config import ROOT, load_paths
from mffer.data import media
from mffer.data.cremad import VALENCE
from mffer.data.splits import FRACTIONS, N_SPLITS, make_split

RAVDESS_EMOTION = {1: "NEU", 3: "HAP", 4: "SAD", 5: "ANG", 6: "FEA", 7: "DIS"}   # 2 calm, 8 surprised: not used
RAVDESS_ACTOR_OFFSET = 2000
N_FRAMES, N_MFCC, N_FFT, HOP = 100, 13, 1024, 640       # as mffer.data.features (V1)
N_VIDEO, SIZE = 10, 224
YUNET = "models/face_detection_yunet_2023mar.onnx"      # relative to the data directory (sha256 8f2383e4...)


def directory():
    return load_paths().processed / "v2"


def split_directory():
    return ROOT / "splits" / "v2"


def ravdess_index(data) -> pd.DataFrame:
    rows = []
    for path in sorted((data / "RAVDESS" / "video_speech").glob("**/01-01-*.mp4")):
        f = list(map(int, re.match(r"^(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)\.mp4$", path.name).groups()))
        if f[2] not in RAVDESS_EMOTION:
            continue
        emotion, actor = RAVDESS_EMOTION[f[2]], f[6]
        wav = data / "RAVDESS" / "audio_speech" / f"Actor_{actor:02d}" / f"03{path.name[2:-4]}.wav"
        level = "XX" if emotion == "NEU" else {1: "RN", 2: "RS"}[f[3]]
        rows.append(dict(clip_id=f"rav_{path.stem}", actor_id=RAVDESS_ACTOR_OFFSET + actor, sentence=f"RS{f[4]}",
                         displayed_emotion=emotion, displayed_level=level, displayed_valence=VALENCE[emotion],
                         displayed_intensity={"XX": np.nan, "RN": 2 / 3, "RS": 1.0}[level],
                         perceived_vote=np.nan, perceived_emotion=np.nan, perceived_intensity=np.nan,
                         perceived_raters=np.nan, age=np.nan, sex="Male" if actor % 2 else "Female",
                         has_video=True, corpus="ravdess",
                         video=str(path.relative_to(data)), wav=str(wav.relative_to(data))))
    return pd.DataFrame(rows)


def build_index() -> pd.DataFrame:
    paths = load_paths()
    data = paths.raw.parent
    cremad = pd.read_csv(paths.processed / "clips.csv")
    cremad["corpus"] = "cremad"
    cremad["video"] = "CREMA-D/VideoFlash/" + cremad["clip_id"] + ".flv"
    cremad["wav"] = "CREMA-D/AudioWAV/" + cremad["clip_id"] + ".wav"
    ravdess = ravdess_index(data)
    clips = pd.concat([cremad, ravdess[[c for c in cremad.columns if c in ravdess]]], ignore_index=True)
    check(clips)
    return clips


def check(clips: pd.DataFrame) -> None:
    assert clips["clip_id"].is_unique
    assert clips.groupby("corpus").size().to_dict() == {"cremad": 7442, "ravdess": 1056}, clips.corpus.value_counts()
    assert clips.groupby("corpus").actor_id.nunique().to_dict() == {"cremad": 91, "ravdess": 24}
    assert clips[["actor_id", "sex"]].drop_duplicates().actor_id.is_unique
    assert clips.loc[clips.corpus == "ravdess", "displayed_level"].isin(["RN", "RS", "XX"]).all()


def make_splits() -> None:
    clips = pd.read_csv(directory() / "clips.csv")
    ravdess = clips.loc[clips.corpus == "ravdess", ["actor_id", "sex"]].drop_duplicates()
    split_directory().mkdir(parents=True, exist_ok=True)
    for split_id in range(N_SPLITS):
        file = split_directory() / f"split_{split_id}.json"
        if file.exists():
            print(f"{file.name}: exists, kept (splits are frozen)")
            continue
        v1 = json.loads((ROOT / "splits" / f"split_{split_id}.json").read_text())
        rav = make_split(ravdess, seed=split_id)
        split = {part: sorted(v1[part] + rav[part]) for part in FRACTIONS}
        file.write_text(json.dumps({"split_id": split_id, "seed": split_id,
                                    "rule": "V1 split's CREMA-D actors + RAVDESS by mffer.data.splits.make_split",
                                    **split}, indent=1))
        print(f"{file.name}: " + ", ".join(f"{p} {len(v1[p])}+{len(rav[p])}" for p in FRACTIONS) + " actors")


# ---------------------------------------------------------------- features

_aligner = None


def _init_worker(model: str) -> None:
    global _aligner
    _aligner = media.FaceAligner(model, size=SIZE)


def mfcc(signal: np.ndarray) -> tuple[np.ndarray, int]:
    coeffs = librosa.feature.mfcc(y=signal, sr=media.SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP).T
    n = min(len(coeffs), N_FRAMES)
    out = np.zeros((N_FRAMES, N_MFCC), np.float32)
    out[:n] = coeffs[:n]
    return out, n


def extract(args):
    data, video, wav = args
    signal = media.load_audio(data / wav)
    start, end, trimmed = media.speech_interval(signal)
    audio, n = mfcc(signal[int(start * media.SAMPLE_RATE):int(end * media.SAMPLE_RATE)])
    frames, fps = media.decode(data / video)
    picks = media.frame_picks(len(frames), fps, (start, end), N_VIDEO)
    faces, share = _aligner.faces(frames, picks)
    return audio, n, faces, dict(speech_start=start, speech_end=end, trimmed=trimmed, face_share=share,
                                 n_video_frames=len(frames), fps=fps)


def write_features(limit: int | None, workers: int) -> None:
    out = directory()
    clips = pd.read_csv(out / "clips.csv").iloc[:limit]
    data = load_paths().raw.parent
    target = out / ("features.h5" if limit is None else f"features-{limit}.h5")
    partial = target.with_suffix(".partial.h5")
    n, info = len(clips), []
    with h5py.File(partial, "w") as h5:
        h5.create_dataset("clip_id", data=np.array(clips.clip_id.tolist(), dtype="S"))
        audio = h5.create_dataset("audio", (n, N_FRAMES, N_MFCC), np.float32)
        lengths = h5.create_dataset("audio_frames", (n,), np.int16)
        video = h5.create_dataset("video", (n, N_VIDEO, SIZE, SIZE, 3), np.uint8, chunks=(1, N_VIDEO, SIZE, SIZE, 3))
        with Pool(workers, initializer=_init_worker, initargs=(str(data / YUNET),)) as pool:
            jobs = pool.imap(extract, [(data, v, w) for v, w in zip(clips.video, clips.wav, strict=True)], chunksize=4)
            for row, (a, length, v, extra) in enumerate(tqdm(jobs, total=n, desc="V2 features", mininterval=30)):
                audio[row], lengths[row], video[row] = a, length, v
                info.append({"clip_id": clips.clip_id.iloc[row], **extra})
    partial.rename(target)
    table = pd.DataFrame(info)
    table.to_csv(target.with_name(target.stem.replace("features", "media") + ".csv"), index=False)
    print(f"wrote {target}")
    print(table.assign(corpus=clips.corpus.to_numpy()).groupby("corpus")[["trimmed", "face_share"]].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["index", "splits", "features"])
    parser.add_argument("--limit", type=int, help="features: only the first N clips (a check file)")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    if args.step == "index":
        directory().mkdir(parents=True, exist_ok=True)
        clips = build_index()
        clips.to_csv(directory() / "clips.csv", index=False)
        print(f"wrote {directory() / 'clips.csv'}: {clips.groupby('corpus').size().to_dict()} clips")
    elif args.step == "splits":
        make_splits()
    else:
        write_features(args.limit, args.workers)


if __name__ == "__main__":
    main()
