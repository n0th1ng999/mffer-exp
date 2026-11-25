# 0025. Model B: Mocanu et al. (2023) on frozen pretrained encoders

- **Status:** Accepted, 25 Nov 2026
- **Where:** `scripts/extract_mocanu.py` (PyTorch env `mffer-torch`: frozen encoders -> `mocanu.h5`),
  `mffer/models.py` (`build_mocanu`, `CrossModalFusion`, `EmotionMetricLoss`, preset `mocanu`),
  `mffer/data/pipeline.py` (`Cremad.read_mocanu`), `mffer/metrics.py` (valence from emotion)

## Context

T1 asks whether a published state-of-the-art audio-visual architecture beats our CNN (A) under our protocol,
and whether it can be used in federated learning. The chosen paper (Mocanu, Tapu & Zaharia, Image and Vision
Computing 133:104676, 2023; no public code) reports 84.57% accuracy on CREMA-D and 89.25% on RAVDESS with
actor-disjoint 10-fold CV. It fine-tunes a Kinetics-pretrained 3D ResNet-101 on 96 frames per clip and an
ImageNet ResNet-18 on spectrograms: 95 min per epoch on an RTX 3090 Ti, ~100 M parameters. Fine-tuning that
is ~240 GPU-hours per run here and ~400 MB per client per FL round: not feasible before 6 Dec 2026.

## Decision

Model B = the paper's architecture on **frozen** pretrained encoders, whose outputs are computed once:

- Encoders (frozen): 3D ResNet-101 (Hara et al. 2018, `r3d101_K_200ep.pth`, Kinetics-700; the paper says
  ImageNet + Kinetics), conv5 of each 16-frame snippet (2048 x 7 x 7); ResNet-18 (torchvision ImageNet) on
  six 1-s spectrogram windows (Hamming 32 ms, hop 10 ms, 256 bins, dB), conv5 averaged over space (512).
- Trained on top, as in the paper: spatial, channel and temporal multi-head self-attention on video (H = 8),
  temporal on audio, the A-V cross-attention fusion (Eq. 7-11, Xavier-initialised W, tanh skip), fully
  connected layers, and the emotion-metric triplet loss with polarity constraint (Eq. 14-16).
- Single task (emotion), as in the paper. Valence for the scores is read off the predicted emotion (exact:
  our valence is a function of the emotion; `mffer.metrics.predictions_table`).
- Same data contract (ADR 0024), splits, optimiser (Adam 1e-4), batch 16 and protocol (ADR 0026) as A.

Choices the paper leaves open (each recorded here so the reimplementation can be checked):

| Gap in the paper | Our choice |
|---|---|
| conv5 size "16 x 16 x 1024" (impossible for 224 px) | 7 x 7 x 2048, linear projection to m = 1024 |
| Six 1-s audio windows from ~2-3 s of speech | windows spread evenly over the speech, overlapping |
| Spectrogram image size and scaling | 256 bins x 100 frames, dB scaled to [0, 1], resized to 224 x 224, ImageNet mean/std |
| Channel attention dimensions (s/H = 49/8) | key dim 7 per head, output 49 |
| "Column-wise softmax" in the fusion | softmax over the rows of each column, applied literally (Eq. 8) |
| Fully connected layers | dropout 0.3, FC 256 (ReLU, = the embedding), softmax 6 |
| Loss combination, margins, mining | cross-entropy + metric loss (weight 1), alpha = beta = 0.2 on squared distances of L2-normalised embeddings, all valid triplets in the batch |
| Related emotions among 6 classes | polarity groups = our valence classes; happy and neutral anchors use the plain triplet term |
| Augmentation (crop, flip, brightness) | none: features are computed once (frozen encoders) |
| Early stopping on accuracy, patience 20 | our protocol 4 (ADR 0026) |

## Consequences

- B is "Mocanu et al.'s architecture with frozen encoders", not a replication of their numbers; the paper must
  say so and cite their protocol differences (fine-tuning, 10-fold CV, 8 RAVDESS emotions, per-corpus models).
- Trained parameters: 12.9 M (A: 0.30 M). In FL only these are exchanged; each kiosk must also run the frozen
  encoders (~95 M parameters, inference only): T6 reports both.
- Pretrained encoders are an exception to the "this dataset only" rule (decided 25 Nov 2026 for B only).
  Kinetics and ImageNet weights carry their own licences and biases.
- No augmentation for B is a disadvantage against A only if A uses augmentation; A's preset (D2) does not.
