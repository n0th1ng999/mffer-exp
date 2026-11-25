"""T0 media preprocessing shared by every V2 feature extractor (ADR 0024). Imports only cv2, numpy and librosa,
so the TensorFlow features (mffer.data.v2) and the PyTorch features of model B (scripts/extract_mocanu.py)
see exactly the same speech interval and the same faces.

- Speech interval: librosa.effects.trim at 30 dB below the peak, on the 16 kHz signal (RAVDESS files are about
  half silence, CREMA-D much less: EDA §6). Audio features and video frames are both taken inside it.
- Faces: YuNet (OpenCV) on every frame used; the largest face's 5 landmarks are smoothed over time (median of
  5 neighbouring frames), and a similarity transform maps them onto a fixed template, so every face has the same
  position, size and rotation and little background is left (EDA §8g, kiosk mitigations 1-2). Frames without a
  detection reuse the nearest detected landmarks; a clip without any detection falls back to a centre crop.
- Lighting: CLAHE on the L channel of Lab (mitigation 3), so colour is kept for the pretrained video network.
"""
from pathlib import Path

import cv2
import librosa
import numpy as np

SAMPLE_RATE = 16_000
TOP_DB = 30
MIN_SPEECH = 0.5     # s; a shorter detected interval (or a broken file) falls back to the whole clip
DETECT_SIDE = 640    # frames are downscaled so the longer side is at most this for detection
# ArcFace's 5-point template for 112 x 112 (eyes, nose tip, mouth corners), loosened by TEMPLATE_SCALE so the
# aligned crop keeps the forehead and the chin (eyes at ~46% height, mouth at ~76%).
ARCFACE_112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)
TEMPLATE_SCALE = 0.85


def load_audio(wav: Path | str) -> np.ndarray:
    signal, _ = librosa.load(wav, sr=SAMPLE_RATE)
    return signal


def speech_interval(signal: np.ndarray) -> tuple[float, float, bool]:
    """(start s, end s, trimmed) of the speech; the whole signal when no interval of MIN_SPEECH is found."""
    length = len(signal) / SAMPLE_RATE
    if len(signal) < SAMPLE_RATE * MIN_SPEECH or not np.any(signal):
        return 0.0, length, False
    _, (start, end) = librosa.effects.trim(signal, top_db=TOP_DB)
    if (end - start) / SAMPLE_RATE < MIN_SPEECH:
        return 0.0, length, False
    return start / SAMPLE_RATE, end / SAMPLE_RATE, True


def decode(path: Path | str) -> tuple[list[np.ndarray], float]:
    """All frames (BGR) and the frame rate."""
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return frames, fps


def frame_picks(n_frames: int, fps: float, interval: tuple[float, float], count: int) -> np.ndarray:
    """`count` frame indices spread uniformly over the speech interval (repeats when it has fewer frames)."""
    first = int(np.clip(round(interval[0] * fps), 0, n_frames - 1))
    last = int(np.clip(round(interval[1] * fps), first, n_frames - 1))
    return np.linspace(first, last, count).round().astype(int)


def template(size: int) -> np.ndarray:
    return ((ARCFACE_112 - 56.0) * TEMPLATE_SCALE * size / 112.0 + size / 2.0).astype(np.float32)


class FaceAligner:
    def __init__(self, model: Path | str, size: int = 224, score: float = 0.6):
        self.model, self.size, self.score = str(model), size, score
        self.detector, self.input_size = None, None
        self.target = template(size)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def landmarks(self, frame: np.ndarray) -> np.ndarray | None:
        """(5, 2) landmarks of the largest face in full-frame pixels, or None."""
        h, w = frame.shape[:2]
        scale = min(1.0, DETECT_SIDE / max(h, w))
        size = (int(w * scale), int(h * scale))
        if self.detector is None or self.input_size != size:
            self.detector = cv2.FaceDetectorYN.create(self.model, "", size, self.score)
            self.input_size = size
        small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if scale < 1 else frame
        _, faces = self.detector.detect(small)
        if faces is None:
            return None
        face = max(faces, key=lambda f: f[2] * f[3])
        return face[4:14].reshape(5, 2) / scale

    def align(self, frame: np.ndarray, points: np.ndarray | None) -> np.ndarray:
        """Aligned, lighting-normalised RGB face (size x size, uint8); centre square when `points` is None."""
        if points is None:
            h, w = frame.shape[:2]
            side = min(h, w)
            top, left = (h - side) // 2, (w - side) // 2
            face = cv2.resize(frame[top:top + side, left:left + side], (self.size, self.size),
                              interpolation=cv2.INTER_AREA)
        else:
            matrix, _ = cv2.estimateAffinePartial2D(points.astype(np.float32), self.target, method=cv2.LMEDS)
            face = cv2.warpAffine(frame, matrix, (self.size, self.size), flags=cv2.INTER_AREA,
                                  borderMode=cv2.BORDER_REPLICATE)
        lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
        lab[..., 0] = self.clahe.apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def faces(self, frames: list[np.ndarray], picks: np.ndarray) -> tuple[np.ndarray, float]:
        """Aligned faces for `picks` (len(picks) x size x size x 3, RGB uint8) and the share of picked frames
        with a detected face."""
        unique = np.unique(picks)
        found = {int(i): self.landmarks(frames[i]) for i in unique}
        detected = np.array([i for i in unique if found[int(i)] is not None])
        if len(detected) == 0:
            return np.stack([self.align(frames[i], None) for i in picks]), 0.0
        raw = np.stack([found[int(detected[np.abs(detected - i).argmin()])] for i in unique])   # nearest detection
        smooth = np.stack([np.median(raw[max(0, k - 2):k + 3], axis=0) for k in range(len(unique))])
        by_frame = dict(zip(unique.tolist(), smooth, strict=True))
        share = float(np.mean([found[int(i)] is not None for i in picks]))
        return np.stack([self.align(frames[i], by_frame[int(i)]) for i in picks]), share
