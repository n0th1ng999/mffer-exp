"""Builds notebooks/eda_ravdess.ipynb section by section (scaffolding; the notebook is the deliverable).

    python scripts/eda_ravdess/build.py && jupyter nbconvert --execute --inplace notebooks/eda_ravdess.ipynb
"""
import sys
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "eda_ravdess.ipynb"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
md("""# RAVDESS: exploratory data analysis

Can RAVDESS join CREMA-D in this project, as a third federated client and as a cross-corpus test set? This
notebook looks at what RAVDESS contains, how it was recorded, and which of our five task heads its labels can
feed, always side by side with CREMA-D (`notebooks/eda.ipynb`, same section style).

Source: Livingstone & Russo (2018), *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)*,
PLoS ONE 13(5): e0196391. Data: Zenodo record 1188976 (DOI 10.5281/zenodo.1188976), licence **CC BY-NC-SA 4.0**
(non-commercial: fine for the paper, not for a product). Stored outside the repo in `../data/RAVDESS/`.

**Contents**
1. What was downloaded, and is it complete
2. What a clip is: the file-name code
3. How the recordings were organised (the grid)
4. Calm and surprised: does calm belong with neutral? (decision: drop both)
5. How far apart are the corpora? Non-IID evidence for federated learning
6. Audio length, silence and the 4-second window
7. Discussion: what this means for training, and how to treat the data
8. Video: what the model sees from each corpus (kiosk view)
9. Zero-shot: do our trained CREMA-D models work on RAVDESS?
10. Where does the visual encoder look? (Grad-CAM)
"""),
code("""import re
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from mffer.config import load_paths
from mffer.data.cremad import VALENCE as CREMAD_VALENCE

paths = load_paths()
clips_cd = pd.read_csv(paths.processed / "clips.csv")            # CREMA-D, for comparison
RAV = paths.raw.parent / "RAVDESS"

# Same chart style as notebooks/eda.ipynb.
BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#52514e"
plt.rcParams.update({
    "figure.dpi": 110, "figure.figsize": (8, 3.2), "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#bdbcb6", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#ecebe7", "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "normal", "axes.titlelocation": "left",
})"""),
md("""## 1. What was downloaded, and is it complete

Zenodo splits RAVDESS into 49 zip files (25.6 GB): audio-only speech and song (one zip each) and one
audio-visual zip per actor for speech and for song. Each zip was checked against Zenodo's MD5 before use.
The paper reports **1,440 speech files per modality**: 24 actors × 60 recordings."""),
code("""files = sorted(RAV.glob("audio_speech/Actor_*/*.wav"))
print(f"audio speech files: {len(files)}  (paper: 1440)")
print(f"actor folders: {len({f.parent.name for f in files})}  (paper: 24)")
print("examples:", *[f.name for f in files[:3]])"""),
md("""## 2. What a clip is: the file-name code

Unlike CREMA-D (`1001_ITS_HAP_XX`), a RAVDESS file name is **seven two-digit fields**, and every label is in it:

| Field | Values |
|---|---|
| modality | 01 audio-video, 02 video only, 03 audio only |
| vocal channel | 01 speech, 02 song |
| emotion | 01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fearful, 07 disgust, 08 surprised |
| intensity | 01 normal, 02 strong (neutral has no strong) |
| statement | 01 "Kids are talking by the door", 02 "Dogs are sitting by the door" |
| repetition | 01 first, 02 second take |
| actor | 01–24; odd = male, even = female |

So `03-01-06-01-02-01-12.wav` = audio only, speech, fearful, normal intensity, "Dogs…", first take, actor 12
(female). The clip table below is built from the names alone."""),
code("""FIELDS = ["modality", "channel", "emotion", "intensity", "statement", "repetition", "actor"]
EMOTION = {1: "neutral", 2: "calm", 3: "happy", 4: "sad", 5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"}
NAME = re.compile(r"^(\\d\\d)-(\\d\\d)-(\\d\\d)-(\\d\\d)-(\\d\\d)-(\\d\\d)-(\\d\\d)\\.wav$")

rows = []
for f in files:
    m = NAME.match(f.name)
    assert m, f"unexpected file name {f.name}"
    rows.append(dict(zip(FIELDS, map(int, m.groups())), file=f.name))
rav = pd.DataFrame(rows)
assert (rav.modality == 3).all() and (rav.channel == 1).all()          # audio-only speech, as downloaded
assert (rav.actor.astype(str).str.zfill(2) == [f.parent.name[-2:] for f in files]).all()  # name matches folder
rav["emotion"] = rav.emotion.map(EMOTION)
rav["intensity"] = rav.intensity.map({1: "normal", 2: "strong"})
rav["sex"] = np.where(rav.actor % 2 == 1, "Male", "Female")
rav.head()"""),
md("""## 3. How the recordings were organised (the grid)

Like CREMA-D, RAVDESS is a **full grid**, but a much denser one per actor and far fewer actors:

- every actor says **2 statements × 2 takes** in each emotion;
- the 7 non-neutral emotions come at **two intensities** (normal, strong); neutral only at normal;
- so 7 × 2 × 2 × 2 + 1 × 2 × 2 = **60 clips per actor**, 24 actors, 1,440 clips.

CREMA-D has 91 actors × ~82 clips over 12 sentences and controls intensity for one sentence only (IEO)."""),
code("""per_actor = rav.groupby("actor").size()
print(f"clips per actor: min {per_actor.min()}, max {per_actor.max()}; actors: {rav.actor.nunique()} "
      f"({(rav.groupby('actor').sex.first() == 'Male').sum()} male, {(rav.groupby('actor').sex.first() == 'Female').sum()} female)")
grid = rav.pivot_table(index="emotion", columns="intensity", values="file", aggfunc="count", fill_value=0)
grid = grid.loc[list(EMOTION.values())]
grid["total"] = grid.sum(axis=1)
grid"""),
code("""compare = pd.DataFrame({
    "CREMA-D": [clips_cd.actor_id.nunique(), len(clips_cd), clips_cd.sentence.nunique(),
                clips_cd.displayed_emotion.nunique(), "IEO only (LO/MD/HI)", round(len(clips_cd) / clips_cd.actor_id.nunique(), 1)],
    "RAVDESS speech": [rav.actor.nunique(), len(rav), rav.statement.nunique(), rav.emotion.nunique(),
                       "every clip (normal/strong)", round(len(rav) / rav.actor.nunique(), 1)],
}, index=["actors", "clips", "sentences", "emotions", "intensity instruction", "clips per actor"])
compare"""),
md("""**Takeaways**
- RAVDESS is complete (1,440 speech clips, 24 actors, 60 each, 12 male / 12 female) and a perfect grid: no
  missing recordings, unlike CREMA-D's 76–81-clip actors.
- It is wide where CREMA-D is narrow: **every** clip carries an intensity instruction (normal/strong), while
  CREMA-D controls intensity on one sentence only. And narrow where CREMA-D is wide: 24 actors instead of 91, 2
  sentences instead of 12. With 60 clips of the same two lines per actor, a model can learn the speaker.
- Every line is recorded **twice** (two takes): near-duplicates. Splits must keep each actor in one part.
- Neutral is half-sized by design (no strong version): 96 clips against 192 for every other emotion."""),
md("""## 4. Calm and surprised: does calm belong with neutral?

RAVDESS has two emotions CREMA-D lacks. Our heads predict CREMA-D's six and derive valence from them, so each
extra emotion must be mapped or dropped. **Surprised** (valence ambiguous: pleasant or not) is dropped in every
option considered. **Calm** is the open question: merging it into neutral (1,248 clips, neutral 23%) is common
in the speech-emotion literature, but calm was acted as a distinct, low-arousal, mildly positive state.

This section puts calm next to neutral and three anchors (happy, sad, and the other emotions) three ways:
**listen**, **look** (spectrograms, acoustic descriptors, a 2-D map), and **test** (can a simple classifier tell
them apart on actors it never saw?). Faces follow in the video section.

Every descriptor is **normalised within actor** (z-score over that actor's 60 clips): voices differ far more
between people than between one person's emotions, and with only 24 actors, raw values mostly measure who is
speaking."""),
code("""import librosa, librosa.display
from IPython.display import Audio, HTML, display

SR = 16000                                               # CREMA-D's rate; RAVDESS files are 48 kHz
CACHE = RAV / "audio_descriptors.csv"

def describe(path):
    y, _ = librosa.load(path, sr=SR)
    if len(y) < SR // 4:                                 # broken recording (< 0.25 s): described as missing
        return {}
    speech, (start, end) = librosa.effects.trim(y, top_db=30)   # cut the leading/trailing silence
    rms = librosa.feature.rms(y=speech, frame_length=400, hop_length=160)[0]
    f0 = librosa.yin(speech, fmin=65, fmax=500, sr=SR, frame_length=1024, hop_length=160)
    n = min(len(f0), len(rms)); voiced = rms[:n] > np.percentile(rms[:n], 40)   # pitch on the louder frames only
    semitones = 12 * np.log2(f0[:n][voiced] / 100)
    if len(semitones) == 0:                              # (near-)silent file: no pitch to measure
        semitones = np.array([np.nan, np.nan])
    mfcc = librosa.feature.mfcc(y=speech, sr=SR, n_mfcc=13, n_fft=400, hop_length=160)
    return dict(duration_s=len(y) / SR, speech_s=len(speech) / SR, lead_s=start / SR, trail_s=(len(y) - end) / SR,
                loudness_db=20 * np.log10(rms.mean() + 1e-9),
                loudness_spread_db=20 * np.log10(np.percentile(rms, 95) + 1e-9) - 20 * np.log10(np.percentile(rms, 20) + 1e-9),
                pitch_st=np.median(semitones), pitch_range_st=np.subtract(*np.percentile(semitones, [90, 10])),
                **{f"mfcc{i}_mean": v for i, v in enumerate(mfcc.mean(1))},
                **{f"mfcc{i}_std": v for i, v in enumerate(mfcc.std(1))})

if CACHE.exists():
    desc = pd.read_csv(CACHE)
else:
    from joblib import Parallel, delayed
    desc = pd.DataFrame(Parallel(n_jobs=8)(delayed(describe)(RAV / "audio_speech" / f"Actor_{a:02d}" / f)
                                           for a, f in zip(rav.actor, rav.file)))
    desc.insert(0, "file", rav.file.values)
    desc.to_csv(CACHE, index=False)
assert (desc.file == rav.file).all()
FEATURES = [c for c in desc.columns if c != "file"]
z = desc[FEATURES].groupby(rav.actor).transform(lambda s: (s - s.mean()) / s.std())   # within-actor z-scores
rav["group"] = np.where(rav.emotion == "calm", "calm " + rav.intensity, rav.emotion)
print(f"{len(desc)} clips described; speech length {desc.speech_s.median():.2f} s median "
      f"(file {desc.duration_s.median():.2f} s)")"""),
md("""### 4a. Listen

The same actor and line (*"Kids are talking by the door"*, first take) in five states. One male (actor 1) and
one female (actor 2) voice."""),
code("""ORDER = ["neutral", "calm normal", "calm strong", "happy", "sad"]

def clip(actor, group, statement=1, take=1):
    emotion, _, intensity = group.partition(" ")
    hit = rav[(rav.actor == actor) & (rav.emotion == emotion) & (rav.statement == statement) & (rav.repetition == take)
              & (rav.intensity == (intensity or "normal"))]
    return RAV / "audio_speech" / f"Actor_{actor:02d}" / hit.file.iloc[0]

for actor in (1, 2):
    cells = []
    for group in ORDER:
        y, _ = librosa.load(clip(actor, group), sr=SR)
        cells.append(f"<td style='text-align:center'>{group}<br>{Audio(y, rate=SR)._repr_html_()}</td>")
    display(HTML(f"<b>Actor {actor} ({'male' if actor % 2 else 'female'})</b><table><tr>{''.join(cells)}</tr></table>"))"""),
md("""### 4b. Look: spectrograms

Log-mel spectrograms (64 bands, 10 ms hop: the log-mel input of ADR 0020), same clips as above. Brighter =
louder. Calm-as-neutral would show the same flat energy and pitch contour as neutral; calm-as-its-own-state
would look softer and slower than neutral, or show happy's movement at lower energy."""),
code("""fig, axes = plt.subplots(2, 5, figsize=(13, 4.6), sharey=True)
for r, actor in enumerate((1, 2)):
    for c, group in enumerate(ORDER):
        y, _ = librosa.load(clip(actor, group), sr=SR)
        y, _ = librosa.effects.trim(y, top_db=30)
        mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_fft=400, hop_length=160, n_mels=64), ref=1.0)
        librosa.display.specshow(mel, sr=SR, hop_length=160, x_axis="time", y_axis="mel", ax=axes[r, c],
                                 vmin=-80, vmax=0, cmap="magma")
        axes[r, c].set_title(f"actor {actor}: {group}", fontsize=9); axes[r, c].grid(False)
        if c: axes[r, c].set_ylabel("")
plt.tight_layout(); plt.show()"""),
md("""### 4c. Look: acoustic descriptors over all 24 actors

Each box is 96 clips (neutral, calm normal, calm strong) or 192 (happy, sad), within-actor z-scores: 0 = that
actor's average delivery. Loudness and pitch are the classic arousal cues; speech length is a tempo cue (same
words, so longer = slower)."""),
code("""SHOW = {"loudness_db": "loudness", "pitch_st": "pitch (median)", "pitch_range_st": "pitch range", "speech_s": "speech length"}
groups = ["neutral", "calm normal", "calm strong", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
colors = {"neutral": BLUE, "calm normal": ORANGE, "calm strong": ORANGE}
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
for ax, (feature, label) in zip(axes, SHOW.items()):
    data = [z.loc[rav.group == g, feature] for g in groups]
    box = ax.boxplot(data, orientation="horizontal", widths=0.6, showfliers=False, patch_artist=True, medianprops={"color": "black"})
    for patch, g in zip(box["boxes"], groups):
        patch.set_facecolor(colors.get(g, "#dcdad4"))
    ax.set_yticks(range(1, len(groups) + 1), groups); ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_title(label); ax.set_xlabel("z within actor"); ax.grid(axis="x", color="#ecebe7"); ax.grid(axis="y", visible=False)
ax.invert_yaxis(); plt.tight_layout(); plt.show()
z.assign(group=rav.group).groupby("group")[list(SHOW)].median().loc[groups].round(2)"""),
md("""### 4d. Look: one map of all clips

Every clip's 26 MFCC statistics (mean and spread of 13 coefficients, within-actor z-scores) projected to 2-D
with PCA. Calm and neutral are highlighted; everything else is grey. If calm is "neutral", the two clouds
overlap; if it is its own state, calm sits apart or drifts toward another emotion."""),
code("""from sklearn.decomposition import PCA

MFCC = [c for c in FEATURES if c.startswith("mfcc")]
xy = PCA(2, random_state=0).fit_transform(z[MFCC].fillna(0))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharex=True, sharey=True)
for ax, highlight in zip(axes, [("neutral", "calm normal", "calm strong"), ("neutral", "calm normal", "happy", "sad")]):
    ax.scatter(*xy.T, s=6, color="#dcdad4", label="other")
    palette = {"neutral": BLUE, "calm normal": ORANGE, "calm strong": "#b8401a", "happy": "#3a9a5b", "sad": "#7b52c1"}
    for g in highlight:
        m = (rav.group == g).values
        ax.scatter(*xy[m].T, s=10, color=palette[g], label=g)
    ax.legend(markerscale=2, frameon=False, fontsize=8); ax.set_xlabel("PC 1"); ax.grid(False)
axes[0].set_ylabel("PC 2"); plt.tight_layout(); plt.show()"""),
md("""### 4e. Test: can a classifier tell them apart on unseen actors?

A logistic regression on the 26 MFCC statistics plus loudness, pitch and speech length (all within-actor
z-scores), scored with **6-fold cross-validation by actor** (4 test actors per fold, never seen in training).
Each pair is balanced (strong-intensity clips excluded, since neutral has none), so **50% = indistinguishable**.
Reference pairs show what "clearly different" and "similar" look like."""),
code("""from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X_all = z[MFCC + list(SHOW)].fillna(0)
def separability(a, b):
    m = rav.emotion.isin([a, b]) & (rav.intensity == "normal")
    X, y, g = X_all[m], (rav.emotion[m] == a).astype(int), rav.actor[m]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    scores = cross_val_score(model, X, y, groups=g, cv=GroupKFold(6), scoring="balanced_accuracy")
    return scores.mean(), scores.std()

pairs = [("calm", "neutral"), ("calm", "sad"), ("calm", "happy"), ("neutral", "sad"), ("neutral", "happy"),
         ("happy", "angry"), ("sad", "fearful")]
sep = pd.DataFrame([(f"{a} vs {b}", *separability(a, b)) for a, b in pairs],
                   columns=["pair", "balanced accuracy", "sd over folds"]).set_index("pair")
sep.style.format("{:.1%}").bar(subset=["balanced accuracy"], vmin=0.5, vmax=1.0, color="#cfe0f6")"""),
md("""**Takeaways (calm, audio only)**
- **Calm is not neutral, acoustically.** Calm is quieter than neutral (loudness z −1.1 vs −0.7), equally low in
  pitch, and clearly **slower** (speech length z −0.3, and +0.6 at strong intensity, vs −0.8 for neutral): calm
  sounds like neutral spoken softly and slowly. A classifier separates calm from neutral on unseen actors at
  **76%** balanced accuracy, about as well as neutral from sad (72%).
- **Calm sounds like sad.** Calm vs sad is the hardest pair: **55%**, close to chance. On the 2-D map calm,
  neutral and sad share the same low-arousal corner, while happy sits clearly apart (calm vs happy 92%).
- **Risk for our primary metric.** Merging calm into neutral would put 192 sad-sounding clips into the *neutral*
  valence class, blurring exactly the neutral/negative boundary that valence macro-F1 depends on. The audio
  alone argues for **dropping calm**; the face (video section) decides whether calm looks pleasant-neutral, which
  would make the merge defensible for the visual and fused models."""),
]

exec((Path(__file__).parent / "eda_part2.py").read_text())
exec((Path(__file__).parent / "eda_part3.py").read_text())
exec((Path(__file__).parent / "eda_part4.py").read_text())
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, OUT)
print("wrote", OUT, len(cells), "cells", file=sys.stderr)
