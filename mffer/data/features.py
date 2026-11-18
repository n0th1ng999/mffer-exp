"""Audio and video features for every clip in clips.csv -> `<processed>/cremad.h5`.

Preprocessing (identical for every run):
- audio: 16 kHz WAV -> 13 MFCCs, n_fft 1024, hop 640 samples (40 ms), so 100 frames cover 4 s
  (99% of clips are shorter); zero-padded or truncated to 100 frames; `audio_frames` stores the
  true length. Stored raw; normalised in the input pipeline with train-split statistics.
- video: 10 frames at uniformly spaced positions, centre-cropped to a square (480x360 -> 360x360),
  resized to 224x224, RGB uint8. Normalised (/255) in the input pipeline.

Rows of every HDF5 dataset follow the row order of clips.csv (`clip_id` is stored to check it).
"""
import argparse
from multiprocessing import Pool

import cv2
import h5py
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from mffer.config import load_paths
from mffer.models import AUDIO_SHAPE, VIDEO_SHAPE

N_FRAMES, N_MFCC = AUDIO_SHAPE
N_VIDEO, SIZE = VIDEO_SHAPE[0], VIDEO_SHAPE[1]
SAMPLE_RATE, N_FFT, HOP = 16_000, 1024, 640


def mfcc(wav_path) -> tuple[np.ndarray, int]:
    signal, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
    coeffs = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP).T
    length = min(len(coeffs), N_FRAMES)
    out = np.zeros(AUDIO_SHAPE, np.float32)
    out[:length] = coeffs[:length]
    return out, length


def frames(video_path) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    decoded = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded.append(frame)
    capture.release()
    if not decoded:
        raise ValueError(f"no frames decoded from {video_path}")
    picks = np.linspace(0, len(decoded) - 1, N_VIDEO).round().astype(int)
    out = np.empty(VIDEO_SHAPE, np.uint8)
    for i, index in enumerate(picks):
        frame = decoded[index]
        height, width = frame.shape[:2]
        side = min(height, width)
        top, left = (height - side) // 2, (width - side) // 2
        square = frame[top:top + side, left:left + side]
        out[i] = cv2.cvtColor(cv2.resize(square, (SIZE, SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return out


def extract(args):
    raw, clip_id = args
    audio, length = mfcc(raw / "AudioWAV" / f"{clip_id}.wav")
    return audio, length, frames(raw / "VideoFlash" / f"{clip_id}.flv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="only the first N clips (for a quick check)")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--out", default="cremad.h5")
    args = parser.parse_args()
    paths = load_paths()
    clip_ids = pd.read_csv(paths.processed / "clips.csv")["clip_id"].tolist()[:args.limit]
    target = paths.processed / args.out
    partial = target.with_suffix(".partial.h5")

    n = len(clip_ids)
    with h5py.File(partial, "w") as h5:
        h5.create_dataset("clip_id", data=np.array(clip_ids, dtype="S"))
        audio = h5.create_dataset("audio", (n, *AUDIO_SHAPE), np.float32)
        lengths = h5.create_dataset("audio_frames", (n,), np.int16)
        video = h5.create_dataset("video", (n, *VIDEO_SHAPE), np.uint8, chunks=(1, *VIDEO_SHAPE))
        with Pool(args.workers) as pool:
            jobs = pool.imap(extract, [(paths.raw, c) for c in clip_ids], chunksize=4)
            for row, (a, length, v) in enumerate(tqdm(jobs, total=n, desc="features")):
                audio[row], lengths[row], video[row] = a, length, v
    partial.rename(target)   # only a complete file gets the final name
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
