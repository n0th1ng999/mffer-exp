"""Frozen pretrained features for model B (Mocanu, Tapu & Zaharia 2023, Image Vis. Comput. 133:104676; ADR 0025).
Runs in the separate PyTorch env (`mffer-torch`): the published weights are PyTorch-only. Writes
`<processed>/v2/mocanu.h5`, row order = `<processed>/v2/clips.csv`.

    PYTHONPATH=. /root/miniforge3/envs/mffer-torch/bin/python scripts/extract_mocanu.py [--limit 50]

Per clip, on the T0 media (mffer.data.media: speech interval, aligned faces, CLAHE), as in the paper §4.2:
- video: 96 aligned 224 x 224 faces spread over the speech interval = N 6 snippets x 16 frames, Kinetics
  mean/std (Hara et al.), through 3D ResNet-101 pretrained on Kinetics-700 (Hara et al. 2018,
  github.com/kenshohara/3D-ResNets-PyTorch `r3d101_K_200ep.pth`, MIT licence) without its pooling and
  classifier: conv5 output 2048 x 1 x 7 x 7 per snippet -> stored as (6, 49, 2048) float16.
- audio: spectrogram of the speech (Hamming 32 ms = 512 samples, hop 10 ms, 256 frequency bins, dB, top
  80 dB), cut into N = 6 one-second windows spread evenly over the speech (overlapping when it is shorter
  than 6 s), each resized to a 224 x 224 x 3 image with ImageNet mean/std, through ResNet-18 pretrained on
  ImageNet (torchvision IMAGENET1K_V1) without pooling and classifier: conv5 7 x 7 x 512 averaged over space
  (paper: "spatial average pooling" before the audio temporal attention) -> stored as (6, 512) float16.
The encoders stay frozen (eval mode, no gradients): only the attention, fusion and classifier on top of
these features are trained (mffer.models, arch "mocanu").
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import h5py
import librosa
import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mffer.data import media  # noqa: E402  (no TensorFlow: only cv2, numpy, librosa)

DATA = Path(__file__).resolve().parents[2] / "data"
OUT = DATA / "processed" / "v2"
WEIGHTS = DATA / "models" / "r3d101_K_200ep.pth"
YUNET = DATA / "models" / "face_detection_yunet_2023mar.onnx"
N_SNIPPETS, SNIPPET, SIZE = 6, 16, 224
KINETICS_MEAN, KINETICS_STD = [0.4345, 0.4051, 0.3775], [0.2768, 0.2713, 0.2737]   # Hara et al., mean.py
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
N_FFT, HOP, WINDOW_FRAMES, TOP_DB = 512, 160, 100, 80


# ---------------------------------------------------------------- 3D ResNet (Hara et al., models/resnet.py, MIT)

def conv3x3x3(i, o, stride=1):
    return nn.Conv3d(i, o, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1x1(i, o, stride=1):
    return nn.Conv3d(i, o, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1, self.bn1 = conv1x1x1(in_planes, planes), nn.BatchNorm3d(planes)
        self.conv2, self.bn2 = conv3x3x3(planes, planes, stride), nn.BatchNorm3d(planes)
        self.conv3, self.bn3 = conv1x1x1(planes, planes * 4), nn.BatchNorm3d(planes * 4)
        self.relu, self.downsample = nn.ReLU(inplace=True), downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + (self.downsample(x) if self.downsample is not None else x))


class ResNet3D(nn.Module):
    """ResNet-101 3D, shortcut B, conv1 7x7x7 temporal stride 1, max-pool: Hara et al.'s defaults."""

    def __init__(self, layers=(3, 4, 23, 3), n_classes=700):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1, self.relu = nn.BatchNorm3d(64), nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(2048, n_classes)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * 4:
            downsample = nn.Sequential(conv1x1x1(self.in_planes, planes * 4, stride), nn.BatchNorm3d(planes * 4))
        layers = [Bottleneck(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * 4
        layers += [Bottleneck(self.in_planes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def features(self, x):
        """conv5 output (the paper removes the global average pooling)."""
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


def video_encoder(device) -> ResNet3D:
    model = ResNet3D()
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state, strict=True)
    return model.eval().to(device)


def audio_encoder(device) -> nn.Module:
    resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    return nn.Sequential(*list(resnet.children())[:-2]).eval().to(device)   # up to conv5 (layer4)


# ---------------------------------------------------------------- inputs (CPU workers)

def spectrogram_windows(signal: np.ndarray) -> np.ndarray:
    """(6, 3, 224, 224) float32 spectrogram images, ImageNet-normalised."""
    power = np.abs(librosa.stft(signal, n_fft=N_FFT, hop_length=HOP, window="hamming")) ** 2
    db = librosa.power_to_db(power[:256], ref=np.max, top_db=TOP_DB)          # 256 frequency components
    image = (db + TOP_DB) / TOP_DB                                            # [0, 1]
    if image.shape[1] < WINDOW_FRAMES:
        image = np.pad(image, ((0, 0), (0, WINDOW_FRAMES - image.shape[1])))  # silence = 0
    starts = np.linspace(0, image.shape[1] - WINDOW_FRAMES, N_SNIPPETS).round().astype(int)
    out = np.empty((N_SNIPPETS, 3, SIZE, SIZE), np.float32)
    for k, s in enumerate(starts):
        window = cv2.resize(image[::-1, s:s + WINDOW_FRAMES], (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        for c in range(3):
            out[k, c] = (window - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    return out


class Clips(torch.utils.data.Dataset):
    def __init__(self, clips: pd.DataFrame):
        self.clips, self.aligner = clips, None

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        if self.aligner is None:                                              # one detector per worker
            self.aligner = media.FaceAligner(YUNET, size=SIZE)
        row = self.clips.iloc[i]
        signal = media.load_audio(DATA / row.wav)
        start, end, _ = media.speech_interval(signal)
        speech = signal[int(start * media.SAMPLE_RATE):int(end * media.SAMPLE_RATE)]
        frames, fps = media.decode(DATA / row.video)
        picks = media.frame_picks(len(frames), fps, (start, end), N_SNIPPETS * SNIPPET)
        faces, share = self.aligner.faces(frames, picks)                      # (96, 224, 224, 3) RGB uint8
        video = torch.from_numpy(faces).reshape(N_SNIPPETS, SNIPPET, SIZE, SIZE, 3).permute(0, 4, 1, 2, 3)
        return i, video.contiguous(), torch.from_numpy(spectrogram_windows(speech)), share


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="only the first N clips (timing check; writes mocanu-<N>.h5)")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch", type=int, default=4, help="clips per GPU batch")
    args = parser.parse_args()
    clips = pd.read_csv(OUT / "clips.csv").iloc[:args.limit]
    device = torch.device("cuda")
    video_net, audio_net = video_encoder(device), audio_encoder(device)
    mean = torch.tensor(KINETICS_MEAN, device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor(KINETICS_STD, device=device).view(1, 3, 1, 1, 1)
    loader = torch.utils.data.DataLoader(Clips(clips), batch_size=args.batch, num_workers=args.workers,
                                         prefetch_factor=2, persistent_workers=False)
    target = OUT / ("mocanu.h5" if args.limit is None else f"mocanu-{args.limit}.h5")
    partial_file, n = target.with_suffix(".partial.h5"), len(clips)
    shares = np.zeros(n, np.float32)
    start = time.perf_counter()
    with h5py.File(partial_file, "w") as h5:
        h5.create_dataset("clip_id", data=np.array(clips.clip_id.tolist(), dtype="S"))
        v_out = h5.create_dataset("video", (n, N_SNIPPETS, 49, 2048), np.float16, chunks=(1, N_SNIPPETS, 49, 2048))
        a_out = h5.create_dataset("audio", (n, N_SNIPPETS, 512), np.float16)
        for done, (index, video, audio, share) in enumerate(loader, start=1):
            b = len(index)
            x = video.to(device).reshape(b * N_SNIPPETS, 3, SNIPPET, SIZE, SIZE).float() / 255.0
            with torch.autocast("cuda", dtype=torch.float16):
                v = video_net.features((x - mean) / std)                      # (b*6, 2048, 1, 7, 7)
                a = audio_net(audio.to(device).reshape(b * N_SNIPPETS, 3, SIZE, SIZE)).mean(dim=(2, 3))
            v = v.float().mean(dim=2).flatten(2).transpose(1, 2).reshape(b, N_SNIPPETS, 49, 2048)
            rows = index.numpy()
            v_out[rows[0]:rows[-1] + 1] = v.cpu().numpy().astype(np.float16)
            a_out[rows[0]:rows[-1] + 1] = a.float().reshape(b, N_SNIPPETS, 512).cpu().numpy().astype(np.float16)
            shares[rows] = share.numpy()
            if done % 50 == 0 or rows[-1] == n - 1:
                rate = (rows[-1] + 1) / (time.perf_counter() - start)
                print(f"{rows[-1] + 1}/{n} clips, {rate:.1f} clips/s, eta {(n - rows[-1] - 1) / rate / 60:.0f} min",
                      flush=True)
        h5.create_dataset("face_share", data=shares)
    partial_file.rename(target)
    print(f"wrote {target}: {(time.perf_counter() - start) / 60:.1f} min, face found on {shares.mean():.1%} of frames")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork", force=True)
    main()
