# ruff: noqa: F821  -- fragment exec()d by build.py, which defines cells, md, code and the imports
# Sections 9-10 of notebooks/eda_ravdess.ipynb: results of scripts/ravdess_probe.py (zero-shot, Grad-CAM).
cells += [
md("""## 9. Zero-shot: do our trained CREMA-D models work on RAVDESS?

The trained networks themselves (not linear probes) are run on RAVDESS's 6-emotion clips, without retraining.
RAVDESS is only a test set here; CREMA-D's test actors are not touched (the CREMA-D column is each run's own,
already-scored test result). Produced by `scripts/ravdess_probe.py zero-shot`; each step adds one change on the
RAVDESS side only:

| Step | Change |
|---|---|
| A | as is: whole-file MFCC with the run's CREMA-D audio statistics, frames over the whole file, centre crop |
| B | + silence trimmed (§6) |
| C | + RAVDESS's own audio statistics (per-client normalisation, §7c) |
| D | + frames sampled inside the speech (§8e) |
| E | + CREMA-D-like framing: face zoomed out to CREMA-D's size and position (§8c) |

Models: U1 (audio only) and C1 (audio + simple visual), protocol 3, splits 0–2; **D2** (audio + deep visual,
gated: our best CREMA-D model) is the protocol-2 split-0 run as a preview until its protocol-3 run exists.
Chance-level valence macro-F1 (always "negative") is 0.27."""),
code("""PROBE = RAV / "probe"
zs = pd.read_csv(PROBE / "zero_shot.csv")
zs["model"] = zs.config.str.split("_").str[0].str.upper()
n_probe = len(pd.read_csv(PROBE / "clips.csv"))
print(f"RAVDESS clips in the probe: {n_probe} of {len(rav6)}")
summary = zs.groupby(["model", "variant"]).agg(
    runs=("run", "nunique"), valence_f1=("valence_macro_f1", "mean"), emotion_f1=("emotion_macro_f1", "mean"),
    intensity_mae=("intensity_mae", "mean"), says_negative=("predicted_negative_share", "mean"),
    cremad_test_valence_f1=("cremad_test_valence_macro_f1", "mean"))
summary.style.format({c: "{:.3f}" for c in ["valence_f1", "emotion_f1", "intensity_mae", "cremad_test_valence_f1"]}
                     | {"says_negative": "{:.0%}"})"""),
code("""fig, ax = plt.subplots(figsize=(9, 3.2))
colors = {"U1": "#7a7974", "C1": BLUE, "D2": ORANGE}
for model, g in zs.groupby("model"):
    m = g.groupby("variant").valence_macro_f1.mean()
    ax.plot(m.index, m.values, marker="o", color=colors.get(model, MUTED), label=f"{model} on RAVDESS")
    ax.axhline(g.cremad_test_valence_macro_f1.mean(), color=colors.get(model, MUTED), lw=0.8, ls="--")
ax.axhline(0.27, color="black", lw=0.6, ls=":"); ax.text(4.05, 0.27, " always negative", va="center", fontsize=8)
ax.set_ylabel("valence macro-F1"); ax.set_xlabel("RAVDESS preprocessing step (dashed: same model on CREMA-D test)")
ax.legend(frameon=False, fontsize=8, loc="upper left"); plt.tight_layout(); plt.show()"""),

md("""## 10. Where does the visual encoder look? (Grad-CAM)

Grad-CAM (Selvaraju et al., ICCV 2017) highlights the image regions that drive the predicted valence class, at
the last spatial layer of the visual encoder. Computed on 96 CREMA-D clips from split 0's **validation** actors
and 96 RAVDESS clips, for the three RAVDESS frame variants. *Saliency on face* is the share of the map inside the
YuNet face box; the *focus ratio* divides it by the face's share of the image (1 = no preference for the face,
above 1 = the network looks at the face)."""),
code("""sal = pd.concat([pd.read_csv(f) for f in sorted(PROBE.glob("saliency_*.csv"))])
sal["model"] = sal.run.str.split("_|-").str[0].str.upper()
sal[["model", "corpus", "frames", "saliency_on_face", "face_area", "face_focus_ratio"]].style.format(
    {"saliency_on_face": "{:.0%}", "face_area": "{:.0%}", "face_focus_ratio": "{:.2f}"}).hide(axis="index")"""),
code("""ex = np.load(next(PROBE.glob("saliency_d2-*.npz")))
rows_cam = [("CREMA-D", "as trained"), ("RAVDESS", "whole"), ("RAVDESS", "framed")]
fig, axes = plt.subplots(3, 6, figsize=(13, 6.8))
for r, (corpus, kind) in enumerate(rows_cam):
    frames_, cams = ex[f"{corpus}|{kind}|frames"], ex[f"{corpus}|{kind}|cam"]
    for c in range(6):
        axes[r, c].imshow(frames_[c]); axes[r, c].imshow(cams[c] / (cams[c].max() + 1e-9), cmap="jet", alpha=0.45)
        axes[r, c].axis("off")
    axes[r, 0].set_title(f"D2: {corpus} ({kind})", fontsize=9)
plt.tight_layout(); plt.show()"""),
md("""**Takeaways (zero-shot and saliency)** (all 24 RAVDESS actors; D2 is the protocol-2 split-0 run)
- **Audio transfers partly, and the §7c treatments recover a third of the gap.** U1 and C1 fall from ~0.60 on
  CREMA-D to ~0.33 on RAVDESS as is, close to always-negative. Trimming silence and normalising with RAVDESS's own
  statistics bring them to **0.44 (C1) and 0.45 (U1)**. Per-client normalisation is the bigger step for emotion F1 (0.24 → 0.37).
  C1 behaves exactly like U1: its simple visual encoder contributes nothing (as on CREMA-D).
- **The best CREMA-D model fails completely.** D2 (0.75 on CREMA-D) answers "negative" for **98–100%** of
  RAVDESS clips in every variant, including CREMA-D-like framing. Better in-domain, worse out of domain: the
  deep visual stream that makes D2 strong on CREMA-D is what breaks on RAVDESS.
- **Grad-CAM shows why.** On CREMA-D, D2 looks at the face: 42% of its saliency falls on a face that covers 16%
  of the image (focus ratio **2.6**), mostly mouth and eyes. On RAVDESS the focus ratio drops to **1.0–1.2**: the
  map spreads over hair edges, the shirt and the background, with or without CREMA-D-like framing. The face is
  where it should be and the right size, but the encoder no longer finds it: the surrounding colours (white
  background, black clothing instead of green screen and grey T-shirts) change the activations it learned on.
- **This is the background effect, measured.** It confirms the concern: a visual encoder trained on one
  background does not transfer to another, even when face size and position match. A face crop (§8g) removes most
  of the context; training on two corpora (or many kiosks) with crops is what should teach the encoder to rely on
  the face alone. (A protocol-3 D2 was never trained: the V1 queue was stopped before it; V2 will retrain.)"""),
]
