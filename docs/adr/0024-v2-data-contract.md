# 0024. V2 data contract (T0): CREMA-D + RAVDESS, aligned faces, audio and frames inside the speech

- **Status:** Accepted, 25 Nov 2026
- **Where:** `mffer/data/media.py` (speech interval, frame picks, `FaceAligner`), `mffer/data/v2.py` (index,
  splits, model A's features), `splits/v2/`, `mffer/data/pipeline.py` (`DATASETS`, `Cremad(dataset=...)`),
  `mffer/data/splits.py` (`dataset` argument, partition `corpus`); run configs set `dataset: v2`

## Context

The RAVDESS EDA (`notebooks/eda_ravdess.ipynb`) showed that a model trained on CREMA-D learns the recording
set-up as much as the face: the corpus is identifiable from the image border alone, RAVDESS files are about
half silence (only 4.4 of 10 sampled frames fall inside the speech), and the best CREMA-D model (D2) answers
"negative" for 98-100% of RAVDESS clips; its Grad-CAM face focus falls from 2.6 on CREMA-D to 1.0-1.2 on
RAVDESS (§9-10). The kiosk use case (CAPE) has the same problem: every store has its own background and light,
which cannot be keyed out (green screen) in practice. RAVDESS joins as the second corpus (6 shared emotions,
calm and surprised dropped, EDA §4f), as both a non-IID client and a second test population.

## Decision

Every V2 model (A and B) sees the same inputs, produced by one module (`media.py`, no TF/torch import):

1. **Speech interval**: `librosa.effects.trim` at 30 dB on the 16 kHz signal; whole clip when shorter than
   0.5 s. Audio features and video frames are both taken inside it.
2. **Faces**: YuNet (OpenCV, `face_detection_yunet_2023mar.onnx`) on every frame used; the largest face's 5
   landmarks, smoothed over time (median of 5), are mapped by a similarity transform onto ArcFace's 5-point
   template scaled by 0.85 (forehead and chin kept); no detection -> nearest detected frame's landmarks; none
   in the clip -> centre crop (the share is recorded per clip).
3. **Lighting**: CLAHE (clip 2.0, 8x8 tiles) on the L channel of Lab; colour is kept.
4. **No background replacement or keying** (not available in a store).
5. **Index and splits**: one `clips.csv` with the CREMA-D label schema and a `corpus` column (RAVDESS actor k
   -> id 2000 + k, intensity normal 2/3 / strong 1, perceived labels masked). Split i = V1 split i's CREMA-D
   actors unchanged + RAVDESS's 24 actors by the same sex-stratified 70/10/20 rule with seed i (18/2/4).
6. **Model A's features** (`features.h5`): V1's MFCC (13 x 100 frames, 40 ms) of the speech, 10 aligned faces
   224 x 224 spread over the speech.

## Consequences

- CREMA-D results of V1 and V2 are not comparable (different inputs); V2 reruns its own baseline (T1, A).
- RAVDESS has only 2 validation actors per split (88 clips): its half of the selection score (ADR 0026) is
  noisy; reported as such.
- Perceived-emotion and perceived-intensity heads train on CREMA-D clips only (masked on RAVDESS, ADR 0006).
- The preprocessing is part of the deployed model: a kiosk must run the same detector, alignment and CLAHE.
