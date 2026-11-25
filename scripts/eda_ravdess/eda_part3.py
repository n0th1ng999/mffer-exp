# ruff: noqa: F821  -- fragment exec()d by build.py, which defines cells, md, code and the imports
# Section 8 (video) of notebooks/eda_ravdess.ipynb; exec'd by build_ravdess_eda.py.
cells += [
md("""## 8. Video: what the model sees from each corpus

CAPE's Portuguese use case is a **smart kiosk**: a fixed, roughly frontal camera, one customer at a time, an
uncontrolled background, and data that stays on the device. So the video questions are: do the two corpora put
the face in the same place and at the same size, does the background give the corpus away, and do the ten
sampled frames actually show the person speaking?

Both corpora go through **our exact frame pipeline** (`mffer.data.features.frames`): ten frames spread evenly over
the whole file, a centred square crop, resized to 224×224. For CREMA-D the frames come straight from
`cremad.h5` (the raw `.flv` files are not stored locally); for RAVDESS they are extracted here from the
audio-video `.mp4` files (modality 01) with the same function.

Faces are found with **YuNet** (`cv2.FaceDetectorYN`, the small neural face detector of the OpenCV model zoo,
`face_detection_yunet_2023mar.onnx`, stored in `../data/models/`), run identically on both corpora."""),
code("""import cv2, h5py
from joblib import Parallel, delayed
from mffer.data.features import frames as pipeline_frames

YUNET = paths.raw.parent / "models" / "face_detection_yunet_2023mar.onnx"
VIDEO_CACHE, MID_CACHE = RAV / "video_stats.csv", RAV / "video_mid_frames.npz"

def face_box(rgb, detector):
    _, found = detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if found is None:
        return None
    x, y, w, h = max(found, key=lambda f: f[2] * f[3])[:4]       # largest face; (x, y, w, h)
    x, y = max(x, 0), max(y, 0)
    return np.array([x, y, min(w, 224 - x), min(h, 224 - y)])

def frame_stats(clip_frames):
    \"\"\"Per clip, from the 10 pipeline frames: face box (median over frames), border/background colour,
    brightness, and motion (mean absolute change between consecutive frames, inside the face box).\"\"\"
    detector = cv2.FaceDetectorYN.create(str(YUNET), "", (224, 224), 0.6)
    boxes = [face_box(f, detector) for f in clip_frames]
    found = [b for b in boxes if b is not None]
    border = np.concatenate([clip_frames[:, :12].reshape(-1, 3), clip_frames[:, -12:].reshape(-1, 3),
                             clip_frames[:, :, :12].reshape(-1, 3), clip_frames[:, :, -12:].reshape(-1, 3)]).astype(float)
    out = dict(face_found=len(found) / len(boxes), brightness=clip_frames.mean(),
               border_r=border[:, 0].mean(), border_g=border[:, 1].mean(), border_b=border[:, 2].mean(),
               border_std=border.std())
    if found:
        x, y, w, h = np.median(np.array(found), axis=0)
        out.update(face_size=w / 224, face_cx=(x + w / 2) / 224, face_cy=(y + h / 2) / 224)
        x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + h)
        faces = clip_frames[:, y0:y1, x0:x1].astype(float).mean(axis=3)
        out["face_motion"] = np.abs(np.diff(faces, axis=0)).mean()
        out["face_motion_per_step"] = ",".join(f"{v:.2f}" for v in np.abs(np.diff(faces, axis=0)).mean(axis=(1, 2)))
    return out

def ravdess_clip(file):
    path = next(RAV.glob(f"video_speech/**/{file}"))
    fr = pipeline_frames(path)
    capture = cv2.VideoCapture(str(path))
    meta = dict(width=capture.get(cv2.CAP_PROP_FRAME_WIDTH), height=capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                fps=capture.get(cv2.CAP_PROP_FPS), n_frames=capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return dict(video=file, **meta, **frame_stats(fr)), fr[5]

available = {p.name for p in RAV.glob("video_speech/**/01-01-*.mp4")}
rav6["video"] = "01" + rav6.file.str[2:].str.replace(".wav", ".mp4")
cached = pd.read_csv(VIDEO_CACHE) if VIDEO_CACHE.exists() else pd.DataFrame(columns=["video"])
mids = dict(np.load(MID_CACHE)) if MID_CACHE.exists() else {}
todo = [v for v in rav6.video if v in available and v not in set(cached.video)]
if todo:
    results = Parallel(n_jobs=4)(delayed(ravdess_clip)(v) for v in todo)
    cached = pd.concat([cached, pd.DataFrame([r for r, _ in results])], ignore_index=True)
    mids.update({r["video"]: m for r, m in results})
    cached.to_csv(VIDEO_CACHE, index=False); np.savez_compressed(MID_CACHE, **mids)
vid = rav6.merge(cached, on="video", how="inner")
print(f"RAVDESS video: {len(vid)} of {len(rav6)} clips available ({vid.actor.nunique()} of 24 actors downloaded)")

CD_VIDEO_CACHE = paths.processed / "cremad_video_stats.csv"
if CD_VIDEO_CACHE.exists():
    vid_cd = pd.read_csv(CD_VIDEO_CACHE)
else:
    cd_rows = np.sort(np.random.default_rng(1).choice(np.flatnonzero(clips_cd.has_video), 1056, replace=False))
    with h5py.File(paths.processed / "cremad.h5") as h5:
        stats = [frame_stats(h5["video"][r]) for r in cd_rows]
    vid_cd = pd.DataFrame(stats).assign(row=cd_rows)
    vid_cd.to_csv(CD_VIDEO_CACHE, index=False)
vid_cd = vid_cd.join(clips_cd[["clip_id", "actor_id", "displayed_emotion", "displayed_level"]], on="row")
print(f"CREMA-D video: {len(vid_cd)} clips sampled ({vid_cd.actor_id.nunique()} actors), same statistics")"""),

md("""### 8a. Format"""),
code("""fmt = pd.DataFrame({
    "CREMA-D": ["480 × 360 (4:3)", "30", "≈ 2.5 s (≈ 75 frames)", "360 × 360 → 224", "2.6 → 1.6 px per model px"],
    "RAVDESS": [f"{int(vid.width.median())} × {int(vid.height.median())} (16:9)", f"{vid.fps.median():.2f}",
                f"≈ {(vid.n_frames / vid.fps).median():.1f} s (≈ {int(vid.n_frames.median())} frames)",
                f"{int(vid.height.median())} × {int(vid.height.median())} → 224", f"{vid.height.median() / 224:.1f} px per model px"],
}, index=["resolution", "frames per second", "clip length", "centre square crop", "down-scaling"])
fmt.loc["down-scaling", "CREMA-D"] = f"{360 / 224:.1f} px per model px"
fmt"""),

md("""### 8b. Side by side: the middle frame (frame 6 of 10) for each emotion

Top: CREMA-D actor 1001 (sentence ITS). Bottom: RAVDESS actor 1 (statement 1, first take, normal intensity).
Exactly the 224×224 image the visual encoder receives."""),
code("""EMO6 = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
fig, axes = plt.subplots(2, 6, figsize=(13, 4.6))
with h5py.File(paths.processed / "cremad.h5") as h5:
    for ax, e in zip(axes[0], EMO6):
        r = clips_cd.index[(clips_cd.actor_id == 1001) & (clips_cd.sentence == "ITS") & (clips_cd.displayed_emotion == e)][0]
        ax.imshow(h5["video"][r, 5]); ax.set_title(f"CREMA-D {e}", fontsize=9); ax.axis("off")
for ax, e in zip(axes[1], EMO6):
    hit = vid[(vid.actor == 1) & (vid.emotion6 == e) & (vid.statement == 1) & (vid.repetition == 1) & (vid.intensity == "normal")]
    if len(hit):
        ax.imshow(mids[hit.video.iloc[0]])
    ax.set_title(f"RAVDESS {e}", fontsize=9); ax.axis("off")
plt.tight_layout(); plt.show()"""),

md("""### 8c. Where the face is, and how big

Face width as a share of the 224-pixel frame, and the face centre position, per clip (median over the ten
frames). The visual encoder has no face detector: it sees whatever the centre crop contains, so a face that is
much smaller or placed elsewhere in one corpus is a different input distribution."""),
code("""fig, axes = plt.subplots(1, 3, figsize=(13, 3.0))
for ax, col, label in zip(axes, ["face_size", "face_cx", "face_cy"], ["face width / frame width", "face centre x / width", "face centre y / height"]):
    a, b = vid_cd[col].dropna(), vid[col].dropna()
    bins = np.linspace(0, 1, 50)
    ax.hist(a, bins, density=True, color=BLUE, alpha=0.6, label="CREMA-D"); ax.hist(b, bins, density=True, color=ORANGE, alpha=0.6, label="RAVDESS")
    ax.set_title(label); ax.set_yticks([])
axes[0].legend(frameon=False, fontsize=8); plt.tight_layout(); plt.show()
pd.DataFrame({name: {"face found (share of frames)": d.face_found.mean(), "face width (median)": d.face_size.median(),
                     "face area share of frame": (d.face_size ** 2).median(), "centre x": d.face_cx.median(), "centre y": d.face_cy.median()}
              for name, d in [("CREMA-D", vid_cd), ("RAVDESS", vid)]}).round(2)"""),

md("""### 8d. Background and colour: a visual corpus fingerprint?

The border of every frame (outer 12 pixels) is background. Its colour and texture, plus overall brightness, are
fed to the same corpus classifier as in §5d (actor-grouped folds, balanced classes, 50% = indistinguishable)."""),
code("""COLOR = ["border_r", "border_g", "border_b", "border_std", "brightness"]
fig, axes = plt.subplots(1, 3, figsize=(12, 2.8))
for ax, col, label in zip(axes, ["border_g", "border_std", "brightness"], ["background brightness (green)", "background texture (std)", "whole-frame brightness"]):
    a, b = vid_cd[col], vid[col]
    bins = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 40)
    ax.hist(a, bins, density=True, color=BLUE, alpha=0.6, label="CREMA-D"); ax.hist(b, bins, density=True, color=ORANGE, alpha=0.6, label="RAVDESS")
    ax.set_title(label); ax.set_yticks([])
axes[0].legend(frameon=False, fontsize=8); plt.tight_layout(); plt.show()

n = min(len(vid), len(vid_cd))
Xv = pd.concat([vid_cd[COLOR].iloc[:n], vid[COLOR].iloc[:n]]).fillna(0)
yv = np.r_[np.zeros(n), np.ones(n)]
gv = np.r_[vid_cd.actor_id.values[:n], 10_000 + vid.actor.values[:n]]
folds = min(6, len(np.unique(vid.actor.values[:n])))
s = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)), Xv, yv, groups=gv,
                    cv=GroupKFold(folds), scoring="balanced_accuracy")
print(f"which corpus? from 5 background/brightness numbers: {s.mean():.1%} ± {s.std():.1%} ({folds} actor-grouped folds)")"""),

md("""### 8e. Do the ten frames show the person speaking?

The ten frames are spread over the **whole file**. RAVDESS files are about half silence (§6), so some frames
fall before the actor starts or after they finish: a resting face. The audio timing of the matching audio-only
file (same take) gives the speech interval; CREMA-D files are cut around the speech, so all ten frames are."""),
code("""timing = rav6.set_index("video")[[]].join(desc6.set_index(rav6.video)[["duration_s", "lead_s", "trail_s"]])
pos = np.linspace(0, 1, 10)
in_speech = np.array([((pos * d >= l) & (pos * d <= d - t)) for d, l, t in timing[["duration_s", "lead_s", "trail_s"]].to_numpy()])
share = in_speech.mean(axis=0)
fig, ax = plt.subplots(figsize=(8, 2.6))
ax.bar(np.arange(1, 11), share, color=ORANGE, width=0.6, label="RAVDESS")
ax.plot(np.arange(1, 11), np.ones(10), color=BLUE, lw=2, label="CREMA-D (cut around speech)")
ax.set_xticks(range(1, 11)); ax.set_xlabel("sampled frame"); ax.yaxis.set_major_formatter("{x:.0%}")
ax.set_title("share of clips in which the frame falls inside the speech"); ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()
print(f"RAVDESS: {in_speech.sum(1).mean():.1f} of 10 frames fall inside the speech on average "
      f"(min {in_speech.sum(1).min()}, max {in_speech.sum(1).max()})")"""),

md("""### 8f. Facial movement by emotion and intensity

Mean absolute pixel change between consecutive sampled frames inside the face box. More movement = more
expressive or more talking. For RAVDESS the change is shown per step, so the silent edges (steps 1 and 9) can be
compared with the middle."""),
code("""motion = pd.DataFrame({
    "CREMA-D": vid_cd.groupby("displayed_emotion").face_motion.median(),
    "RAVDESS normal": vid[vid.intensity == "normal"].groupby("emotion6").face_motion.median(),
    "RAVDESS strong": vid[vid.intensity == "strong"].groupby("emotion6").face_motion.median(),
}).loc[EMO6]
fig, axes = plt.subplots(1, 2, figsize=(13, 3.0), gridspec_kw={"width_ratios": [3, 2]})
x = np.arange(len(EMO6)); w = 0.27
for i, (col, color) in enumerate(zip(motion.columns, [BLUE, ORANGE, "#b8401a"])):
    axes[0].bar(x + (i - 1) * w, motion[col], w, color=color, label=col)
axes[0].set_xticks(x, EMO6); axes[0].set_title("face motion (median per clip)"); axes[0].legend(frameon=False, fontsize=8)
steps = np.array([list(map(float, s.split(","))) for s in vid.face_motion_per_step.dropna()])
steps_cd = np.array([list(map(float, s.split(","))) for s in vid_cd.face_motion_per_step.dropna()])
axes[1].plot(range(1, 10), np.median(steps_cd, 0), color=BLUE, marker="o", label="CREMA-D")
axes[1].plot(range(1, 10), np.median(steps, 0), color=ORANGE, marker="o", label="RAVDESS")
axes[1].set_xticks(range(1, 10)); axes[1].set_xlabel("step between sampled frames"); axes[1].set_title("face motion over the clip")
axes[1].legend(frameon=False, fontsize=8); plt.tight_layout(); plt.show()
motion.round(2)"""),
md("""**Takeaways (video)** (all 24 RAVDESS actors)
- **The background names the corpus.** CREMA-D was filmed against a **green screen**, RAVDESS against **white**.
  Five numbers about the frame border identify the corpus **100%** of the time. A visual encoder that sees the
  whole centre crop gets a perfect client label for free, the visual twin of RAVDESS's leading silence (§6).
- **Different framing.** CREMA-D is head-and-shoulders with the face in the upper half (face ≈ 35% of the frame
  width, 12% of its area); RAVDESS is a close-up with the face centred (≈ 46% of the width, 21% of the area). After
  the 224×224 resize a RAVDESS face has ~1.7× the pixels of a CREMA-D face.
- **Most RAVDESS frames show a silent face.** Only **4.4 of the 10** sampled frames fall inside the speech
  (frames 4–7 almost always, the outer ones almost never): the encoder sees a resting face more than half the
  time, while every CREMA-D frame shows the actor speaking. The motion curve agrees: RAVDESS faces move most in
  the middle of the clip.
- **RAVDESS faces move more, and intensity is visible.** Median face motion is higher than CREMA-D's for every
  emotion, and *strong* clearly above *normal* for every emotion (anger 17.6 vs 13.1), the visual counterpart of §5c. Neutral and sad
  move least in both corpora."""),
md("""### 8g. What this means for the kiosk setting and for training

A CAPE kiosk camera will see a customer at arm's length, roughly frontal, in front of **whatever is behind them in
that store**. Each kiosk is a federated client with its own background, lighting and camera, i.e. exactly the
visual feature skew measured above, only stronger. Three treatments follow, all preprocessing (same for every
corpus, so the comparison with the current CREMA-D runs stays clean):

| Issue | If ignored | Treatment |
|---|---|---|
| background identifies the corpus / kiosk (8d) | the model learns the background as a shortcut; it will not transfer to a new store | **crop to the face** with a detector (YuNet), a fixed margin around the box, then resize to 224 |
| face size differs 1.7× (8c) | the encoder must learn two scales of the same face | the same face crop normalises scale and position |
| 56% of RAVDESS frames are silent faces (8e) | the visual stream is diluted with resting faces | sample the 10 frames **inside the speech interval** (from the audio trim of §6), not over the whole file |

Face cropping was out of scope for the CREMA-D-only protocol (one background, one framing, so it could not
matter). With a second corpus, and with CAPE's kiosks in view, it becomes the visual counterpart of trimming
silence and per-client normalisation: the three together remove the recording fingerprints and leave what a
federated model should learn, the emotion."""),
]
