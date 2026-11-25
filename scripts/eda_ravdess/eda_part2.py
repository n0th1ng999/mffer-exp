# ruff: noqa: F821  -- fragment exec()d by build.py, which defines cells, md, code and the imports
# Sections 4f-7 of notebooks/eda_ravdess.ipynb; exec'd by build_ravdess_eda.py (md, code, cells in scope).
cells += [
md("""### 4f. Decision: drop calm and surprised

**Both are dropped** (decided 25 Nov 2026). Neither exists in CREMA-D, so neither can be scored by our heads
without a mapping, and the audio evidence above argues against the only common mapping (calm → neutral). From
here on, `rav6` holds the **1,056 clips of the six shared emotions**. The same mapping as CREMA-D gives valence:
anger, disgust, fear, sad → negative; neutral → neutral; happy → positive."""),
code("""TO_CREMAD = {"angry": "ANG", "disgust": "DIS", "fearful": "FEA", "happy": "HAP", "neutral": "NEU", "sad": "SAD"}
keep = rav.emotion.isin(TO_CREMAD)
rav6, z6, desc6 = rav[keep].reset_index(drop=True), z[keep].reset_index(drop=True), desc[keep].reset_index(drop=True)
rav6["emotion6"] = rav6.emotion.map(TO_CREMAD)
rav6["valence"] = rav6.emotion6.map(CREMAD_VALENCE)
print(f"{len(rav6)} clips kept of {len(rav)} ({len(rav) - len(rav6)} calm/surprised dropped); "
      f"{len(rav6) // rav6.actor.nunique()} per actor")"""),

md("""## 5. How far apart are the corpora? Non-IID evidence for federated learning

If CREMA-D and RAVDESS become two federated clients (or two groups of clients), the data is **non-IID by
construction**. The federated-learning literature separates four kinds of non-IID data; each is measured
below, so the paper can state *how* non-IID the setting is instead of assuming it:

| Kind | Question | Measured in |
|---|---|---|
| Quantity skew | do clients hold very different amounts of data? | 5a |
| Label skew | do clients see a different mix of classes? | 5b |
| Intensity (label definition) | does the same intensity label mean the same thing? | 5c |
| Feature skew | does the same class *sound* different per client? | 5d |
| Concept shift | does a model of one corpus transfer to the other? | 5e |

CREMA-D gets the same descriptors as RAVDESS (same code, 16 kHz, within-actor z-scores over each actor's
clips)."""),
code("""CD_CACHE = paths.processed / "cremad_audio_descriptors.csv"
if CD_CACHE.exists():
    desc_cd = pd.read_csv(CD_CACHE)
else:
    from joblib import Parallel, delayed
    desc_cd = pd.DataFrame(Parallel(n_jobs=8)(delayed(describe)(paths.raw / "AudioWAV" / f"{c}.wav")
                                              for c in clips_cd.clip_id))
    desc_cd.insert(0, "clip_id", clips_cd.clip_id.values)
    desc_cd.to_csv(CD_CACHE, index=False)
assert (desc_cd.clip_id == clips_cd.clip_id).all()
z_cd = desc_cd[FEATURES].groupby(clips_cd.actor_id).transform(lambda s: (s - s.mean()) / s.std())
print(f"CREMA-D: {len(desc_cd)} clips described")"""),
md("""### 5a. Quantity skew"""),
code("""quantity = pd.DataFrame({
    "CREMA-D": [clips_cd.actor_id.nunique(), len(clips_cd), round(len(clips_cd) / clips_cd.actor_id.nunique(), 1)],
    "RAVDESS (6 emotions)": [rav6.actor.nunique(), len(rav6), round(len(rav6) / rav6.actor.nunique(), 1)],
}, index=["actors", "clips", "clips per actor"])
quantity["ratio"] = (quantity.iloc[:, 0] / quantity.iloc[:, 1]).round(1)
quantity"""),
md("""### 5b. Label skew

Share of each emotion and valence class per corpus, and the Jensen–Shannon divergence between the two
distributions (0 = identical, 1 = no overlap; base 2)."""),
code("""from scipy.spatial.distance import jensenshannon

EMO = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]; VAL = ["negative", "neutral", "positive"]
emo = pd.DataFrame({"CREMA-D": clips_cd.displayed_emotion.value_counts(normalize=True),
                    "RAVDESS": rav6.emotion6.value_counts(normalize=True)}).loc[EMO]
val = pd.DataFrame({"CREMA-D": clips_cd.displayed_valence.value_counts(normalize=True),
                    "RAVDESS": rav6.valence.value_counts(normalize=True)}).loc[VAL]

fig, axes = plt.subplots(1, 2, figsize=(12, 3.2), gridspec_kw={"width_ratios": [2, 1]})
for ax, table, title in [(axes[0], emo, "displayed emotion"), (axes[1], val, "displayed valence")]:
    x = np.arange(len(table)); w = 0.38
    ax.bar(x - w / 2, table["CREMA-D"], w, color=BLUE, label="CREMA-D")
    ax.bar(x + w / 2, table["RAVDESS"], w, color=ORANGE, label="RAVDESS")
    ax.set_xticks(x, table.index); ax.set_title(f"{title}: share of clips"); ax.yaxis.set_major_formatter("{x:.0%}")
    js = jensenshannon(table["CREMA-D"], table["RAVDESS"], base=2)
    ax.text(0.99, 0.95, f"JS divergence {js:.3f}", transform=ax.transAxes, ha="right", va="top", color=MUTED)
axes[0].legend(frameon=False); plt.tight_layout(); plt.show()
pd.concat([emo, val]).style.format("{:.1%}")"""),
md("""### 5c. Intensity: does "strong" mean "high"?

CREMA-D's displayed-intensity head is trained only on sentence IEO, at LO / MD / HI (targets 1/3, 2/3, 1).
RAVDESS asks for normal or strong on every non-neutral clip. Before mapping one onto the other, compare how
much each instruction actually changes the voice: loudness and pitch range, as within-actor z-scores (neutral
excluded on both sides)."""),
code("""cd_ieo = (clips_cd.sentence == "IEO") & (clips_cd.displayed_emotion != "NEU") & clips_cd.displayed_level.isin(["LO", "MD", "HI"])
rows_i = [("CREMA-D " + lvl, z_cd.loc[cd_ieo & (clips_cd.displayed_level == lvl)]) for lvl in ("LO", "MD", "HI")]
rows_i += [("RAVDESS " + lvl, z6.loc[(rav6.emotion6 != "NEU") & (rav6.intensity == lvl)]) for lvl in ("normal", "strong")]
INT_FEATS = {"loudness_db": "loudness", "pitch_st": "pitch (median)", "pitch_range_st": "pitch range"}

fig, axes = plt.subplots(1, 3, figsize=(13, 2.9), sharey=True)
for ax, (feature, label) in zip(axes, INT_FEATS.items()):
    box = ax.boxplot([r[feature].dropna() for _, r in rows_i], orientation="horizontal", widths=0.6, showfliers=False,
                     patch_artist=True, medianprops={"color": "black"})
    for patch, (name, _) in zip(box["boxes"], rows_i):
        patch.set_facecolor(BLUE if name.startswith("CREMA") else ORANGE)
    ax.set_yticks(range(1, len(rows_i) + 1), [n for n, _ in rows_i]); ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_title(label); ax.set_xlabel("z within actor"); ax.grid(axis="x", color="#ecebe7"); ax.grid(axis="y", visible=False)
ax.invert_yaxis(); plt.tight_layout(); plt.show()

steps = pd.DataFrame({name: r[list(INT_FEATS)].median() for name, r in rows_i}).T.rename(columns=INT_FEATS)
steps.round(2)"""),
md("""### 5d. Feature skew: can a classifier tell which corpus a clip came from?

Two versions. **Raw** descriptors (as a microphone records them): this measures recording conditions (studio,
microphone, level, language accent). **Within-actor z-scores**: speaker and recording level removed, so what
remains is the *style* of acting. Each is scored with actor-grouped cross-validation (no actor in both train and
test); classes are balanced by subsampling CREMA-D to RAVDESS's size, so 50% = indistinguishable."""),
code("""from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALL = MFCC + list(SHOW)
rng = np.random.default_rng(0)
cd_sample = np.sort(rng.choice(len(clips_cd), len(rav6), replace=False))
def corpus_test(cd_table, rav_table):
    X = pd.concat([cd_table.iloc[cd_sample][ALL], rav_table[ALL]]).fillna(0)
    y = np.r_[np.zeros(len(cd_sample)), np.ones(len(rav6))]
    groups = np.r_[clips_cd.actor_id.values[cd_sample], 10_000 + rav6.actor.values]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
    s = cross_val_score(model, X, y, groups=groups, cv=GroupKFold(6), scoring="balanced_accuracy")
    return f"{s.mean():.1%} ± {s.std():.1%}"

desc_cd6 = desc_cd.copy()
pd.DataFrame({"which corpus?": [corpus_test(desc_cd, desc6), corpus_test(z_cd, z6)]},
             index=["raw descriptors (recording + style)", "within-actor z-scores (style only)"])"""),
md("""Where the corpora differ in raw terms, per descriptor:"""),
code("""RAW = {"loudness_db": "loudness (dB)", "pitch_st": "pitch (st re 100 Hz)", "speech_s": "speech length (s)",
       "lead_s": "leading silence (s)", "trail_s": "trailing silence (s)"}
fig, axes = plt.subplots(1, 5, figsize=(15, 2.6))
for ax, (feature, label) in zip(axes, RAW.items()):
    a, b = desc_cd[feature].dropna(), desc6[feature].dropna()
    bins = np.linspace(min(a.quantile(0.01), b.quantile(0.01)), max(a.quantile(0.99), b.quantile(0.99)), 40)
    ax.hist(a, bins, density=True, color=BLUE, alpha=0.6, label="CREMA-D")
    ax.hist(b, bins, density=True, color=ORANGE, alpha=0.6, label="RAVDESS")
    ax.set_title(label); ax.set_yticks([])
axes[0].legend(frameon=False, fontsize=8); plt.tight_layout(); plt.show()"""),
md("""### 5e. Concept shift: does a model of one corpus work on the other?

A 6-emotion classifier (same features, within-actor z-scores) trained on one corpus and tested on (a) unseen
actors of the same corpus and (b) all of the other corpus. The drop from (a) to (b) is the shift a single
shared model has to absorb. Chance is 1/6 = 16.7%. This is a weak linear model on hand-made features, so the
absolute numbers are low; the *gap* is what matters."""),
code("""def emotion_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))

corpora = {"CREMA-D": (z_cd[ALL].fillna(0), clips_cd.displayed_emotion.values, clips_cd.actor_id.values),
           "RAVDESS": (z6[ALL].fillna(0), rav6.emotion6.values, rav6.actor.values)}
transfer = {}
for train_name, (X, y, g) in corpora.items():
    within = cross_val_score(emotion_model(), X, y, groups=g, cv=GroupKFold(6), scoring="balanced_accuracy").mean()
    test_name = next(n for n in corpora if n != train_name)
    Xt, yt, _ = corpora[test_name]
    from sklearn.metrics import balanced_accuracy_score
    across = balanced_accuracy_score(yt, emotion_model().fit(X, y).predict(Xt))
    transfer[f"trained on {train_name}"] = {"same corpus, unseen actors": within, "other corpus": across,
                                            "drop (points)": 100 * (within - across)}
pd.DataFrame(transfer).T.style.format({"same corpus, unseen actors": "{:.1%}", "other corpus": "{:.1%}",
                                       "drop (points)": "{:+.1f}"})"""),
md("""**Takeaways (non-IID)**
- **Quantity skew: strong.** CREMA-D has 7× the clips and 3.8× the actors. Under FedAvg (weights ∝ clips) a
  CREMA-D client would contribute ~88% of every global update.
- **Label skew: mild.** The two corpora have almost the same emotion mix (JS divergence 0.07); the only real
  difference is neutral (9% in RAVDESS vs 15%), so RAVDESS is slightly *more* negative-heavy (73% vs 68%).
  Corpus-as-client therefore gives a **feature-skew** setting, not a label-skew one; a label-skewed setting would
  have to be made by partitioning emotions or actors on purpose.
- **Intensity: the mapping is clear.** RAVDESS *normal* moves the voice like CREMA-D *MD* (loudness z −0.28 vs
  −0.32), and *strong* like *HI* (+0.87 vs +0.96; pitch +0.77 vs +1.00). So **normal → 2/3, strong → 1** on our
  displayed-intensity scale. CREMA-D's LO has no RAVDESS counterpart.
- **Feature skew: very strong, but almost all of it is recording, not acting.** A linear classifier names the
  corpus of a clip **99.8%** of the time from raw descriptors, and **52.7%** (chance) once each actor's level and
  voice are normalised away. The raw histograms show where it comes from: RAVDESS is quieter and higher-pitched
  on average, its speech is shorter, and every RAVDESS file starts and ends with ~1 s of silence, which CREMA-D's
  files never do.
- **Concept shift: real.** Even on normalised features, a model trained on one corpus loses **8 points**
  (CREMA-D → RAVDESS) and **16 points** (RAVDESS → CREMA-D) on the other. The same emotion is acted differently
  enough that a shared model must absorb it; RAVDESS, with 24 actors, generalises worse."""),

md("""## 6. Audio length, silence and the 4-second window

Our audio inputs cover a fixed **4 s** window from the start of the file (MFCC: 100 frames × 40 ms; log-mel:
400 × 10 ms), zero-padded or cut. CREMA-D's pipeline does **not** trim silence. What would the same pipeline do
to RAVDESS?"""),
code("""WINDOW = 4.0
def window_stats(d):
    speech_end = d.duration_s - d.trail_s
    return pd.Series({
        "file length (median s)": d.duration_s.median(),
        "speech (median s)": d.speech_s.median(),
        "leading silence (median s)": d.lead_s.median(),
        "trailing silence (median s)": d.trail_s.median(),
        "silence share of file": 1 - (d.speech_s / d.duration_s).median(),
        "files longer than 4 s": (d.duration_s > WINDOW).mean(),
        "speech cut by the window": (speech_end > WINDOW).mean(),
        # The model masks padding (zeros), not silence: silence frames are real, low-level frames.
        "silence among the real frames (untrimmed)": (1 - np.minimum(d.speech_s, np.clip(WINDOW - d.lead_s, 0, None))
                                                      / np.minimum(d.duration_s, WINDOW)).median(),
        "speech starts at (untrimmed, s)": d.lead_s.median(),
        "padding share of the window (trimmed)": (1 - np.minimum(d.speech_s, WINDOW) / WINDOW).median(),
    })
win = pd.DataFrame({"CREMA-D": window_stats(desc_cd), "RAVDESS": window_stats(desc6)})
SECONDS = [r for r in win.index if "(median s)" in r or r.endswith(", s)")]
win.style.format("{:.0%}").format("{:.2f}", subset=pd.IndexSlice[SECONDS, :])"""),
code("""fig, ax = plt.subplots(figsize=(9, 2.6))
for i, (name, d, color) in enumerate([("CREMA-D", desc_cd, BLUE), ("RAVDESS", desc6, ORANGE)]):
    lead, speech, trail = d.lead_s.median(), d.speech_s.median(), d.trail_s.median()
    ax.barh(i, lead, color="#dcdad4"); ax.barh(i, speech, left=lead, color=color); ax.barh(i, trail, left=lead + speech, color="#dcdad4")
    ax.text(lead + speech / 2, i, f"speech {speech:.1f} s", ha="center", va="center", color="white", fontsize=9)
ax.axvline(WINDOW, color="black", lw=1); ax.text(WINDOW, 1.45, " 4 s model window", va="center", fontsize=9)
ax.set_yticks([0, 1], ["CREMA-D", "RAVDESS"]); ax.set_xlabel("seconds (median clip: silence grey)"); ax.grid(axis="y", visible=False)
ax.set_ylim(-0.6, 1.7); plt.tight_layout(); plt.show()"""),
md("""**Takeaways (length)**
- CREMA-D files are cut tightly around the speech (median 0.00 s silence either side); **RAVDESS files are about
  half silence**: ~1.0 s before and ~0.9 s after a 1.8 s utterance.
- The 4 s window is not the problem: after trimming, every RAVDESS utterance fits (median 1.8 s; only 1% of
  speech would be cut, and only without trimming).
- The problem is that **silence is not padding**. The pipeline zero-pads after the file ends, and masked pooling
  (ADR 0018) ignores those zeros, but RAVDESS's leading and trailing silence are real, low-level frames: the
  model would see ~1 s of room noise in front of every RAVDESS clip and never in front of a CREMA-D clip. That is
  a perfect corpus fingerprint, and exactly the kind of shortcut a model takes instead of learning emotion."""),

md("""## 7. Discussion: what this means for training, and how to treat the data

Sections 1–6 turn into concrete choices. Each row says what the data showed, what it would do to training if
ignored, and the proposed treatment. These are proposals for the integration ADR, not yet decisions.

### 7a. Labels: which heads RAVDESS can feed

| Head | RAVDESS has | Treatment |
|---|---|---|
| displayed emotion | 6 shared emotions (calm, surprised dropped: §4) | same 6 classes |
| displayed valence | derived exactly as CREMA-D | same 3 classes |
| displayed intensity | normal / strong on every non-neutral clip | **normal → 2/3, strong → 1** (§5c); neutral masked, as in CREMA-D. Coverage jumps from 18% (CREMA-D) to 91% (RAVDESS) |
| perceived emotion / intensity | no per-clip crowd labels in the Zenodo release | **masked** (weight 0) for RAVDESS clips; the existing mask logic (ADR 0006) already handles a head without labels |

The coverage gap matters: the loss weight of a masked head is divided by its coverage on the training rows
(ADR 0006). In federated training each client sees a different coverage, so the compensation must be computed
**per client** (as class weights already are, ADR 0019), not once for the pooled data.

### 7b. Model inputs: which presets make sense

M3 (C2) uses two inputs RAVDESS cannot fill honestly: **text** is a one-hot of CREMA-D's 12 sentences (RAVDESS
has 2 different ones), and **demographics** needs each actor's age (RAVDESS gives sex only, from the actor
number). Filling them with zeros would make "text = zero" mean "RAVDESS", one more corpus fingerprint.
So cross-corpus experiments should use **D2** (deep visual encoder + gating, audio + video only). It is also our
best model so far, and D4 showed text adds nothing on CREMA-D.

### 7c. Audio: make the corpora differ only where they really differ

| Issue (section) | If ignored | Treatment |
|---|---|---|
| 48 kHz files | features at a different resolution | resample to 16 kHz (the pipeline already loads at 16 kHz) |
| ~1 s silence before and after every RAVDESS clip (§6) | perfect corpus fingerprint; RAVDESS's "speech" is half silence | **trim** leading/trailing silence (top_db 30) for **both** corpora, so one rule applies everywhere |
| level and recording differences (§5d: 99.8% → 52.7%) | the model can tell clients apart from the first frame; a global normaliser leaves the offset in | normalise with **each client's own** statistics (see 7d) |
| shorter speech (1.8 s vs 2.4 s) | more padding in RAVDESS windows | masked pooling (ADR 0018) makes padding irrelevant; worth switching on for multi-corpus runs |

### 7d. Federated learning: what the non-IID evidence implies

The current federated pipeline pools every client's audio statistics into **one global normaliser**
(`AudioStats.combine`, ADR 0008). That was right when every client was CREMA-D; across corpora it keeps the
recording offset of §5d in the inputs. Two options, both testable:

1. **Per-client normalisation**: each client standardises its audio with its own training statistics, and
   nothing about the statistics leaves the client (also a small privacy gain). §5d shows this alone should remove
   most of the corpus difference.
2. **FedBN** (Li et al., ICLR 2021): batch-norm layers stay local, everything else is averaged. Designed for
   exactly this case (feature-skewed clients) and a natural extra experiment next to FedAvg and FedProx (F4).

The quantity skew (7×) means **FedAvg's clip-weighted average lets CREMA-D dominate**. Results must therefore be
reported **per corpus**, never only pooled, or a model that ignores RAVDESS will still look good.

### 7e. Splits and evaluation

- **Actor-disjoint splits within RAVDESS**, like CREMA-D's: 24 actors → 16 train / 4 val / 4 test, each part
  sex-balanced (odd/even actor numbers), three frozen splits. Both takes of a line belong to one actor, so the
  near-duplicates never cross parts.
- The test part is small: 4 actors × 44 clips = 176 clips, about 16 neutral. Valence macro-F1 on that is noisy;
  three splits and per-split reporting are essential.
- **Evaluation grid** for the paper: each model on each corpus's test actors, plus cross-corpus transfer
  (trained on CREMA-D only, tested on RAVDESS, and back) as the "no collaboration" baseline next to local-only (L1).

### 7f. What is still open

- **Video** (next section, once the download finishes): RAVDESS is 1280×720, head-and-shoulders on a white
  background, vs CREMA-D's 480×360. The same fingerprint question applies to the face crop and background.
- Whether RAVDESS's perceptual validation ratings can be obtained per clip (they would un-mask the perceived
  heads).
- Licence: CC BY-NC-SA 4.0 is fine for the paper; any derived model weights inherit the non-commercial terms."""),
]
