"""CREMA-D clip index: one row per clip with actor, sentence, labels and demographics.

This module owns the definition of every label (decided 18 Nov 2026):

- displayed_emotion: the emotion the actor was asked to portray (filename).
- displayed_valence: HAP -> positive, NEU -> neutral, ANG/DIS/FEA/SAD -> negative (derived from the above).
- displayed_intensity: LO/MD/HI -> 1/3, 2/3, 1. Only sentence IEO has these levels; every other clip is
  "XX" (unspecified) and gets NaN, so the head is trained and scored on IEO clips only.
- perceived_emotion: majority of the audio-visual ratings (summaryTable MultiModalVote). The 420 clips
  with a tied vote (e.g. "N:S") get NaN and are masked for this head only.
- perceived_intensity: mean audio-visual rated intensity over all raters / 100 (tabulatedVotes,
  audio-visual rows), defined for every clip including ties.
- sentence: one of 12 scripted sentences; the text input is its one-hot ID.
- age, sex: from VideoDemographics.csv.

Writes `<processed>/clips.csv`.
"""
import argparse
import subprocess

import numpy as np
import pandas as pd

from mffer.config import load_paths

EMOTIONS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
VOTE_TO_EMOTION = dict(zip("ADFHNS", EMOTIONS))
VALENCE = {"HAP": "positive", "NEU": "neutral", "ANG": "negative", "DIS": "negative",
           "FEA": "negative", "SAD": "negative"}
VALENCES = ["negative", "neutral", "positive"]
LEVELS = {"LO": 1 / 3, "MD": 2 / 3, "HI": 1.0}
AUDIOVISUAL_ROWS = 300_000   # tabulatedVotes row ids: 1xxxxx voice, 2xxxxx face, 3xxxxx audio-visual
# The CREMA-D version every result in the paper uses (github.com/CheyneyComputerScience/CREMA-D).
CREMAD_COMMIT = "1658cd342dff90010aa843eaeebd53610a08b1dc"   # 21 Mar 2025


def check_version(raw) -> None:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=raw, capture_output=True, text=True).stdout.strip()
    assert head == CREMAD_COMMIT, (f"CREMA-D at {raw} is at commit {head or '?'}, expected {CREMAD_COMMIT}: "
                                   f"run `git -C {raw} checkout {CREMAD_COMMIT}`")


def build_index(raw) -> pd.DataFrame:
    wavs = sorted((raw / "AudioWAV").glob("*.wav"))
    clips = pd.DataFrame({"clip_id": [w.stem for w in wavs]})
    parts = clips["clip_id"].str.split("_", expand=True)
    clips["actor_id"] = parts[0].astype(int)
    clips["sentence"] = parts[1]
    clips["displayed_emotion"] = parts[2]
    clips["displayed_level"] = parts[3]
    clips["displayed_valence"] = clips["displayed_emotion"].map(VALENCE)
    clips["displayed_intensity"] = clips["displayed_level"].map(LEVELS)   # NaN for XX

    summary = pd.read_csv(raw / "processedResults" / "summaryTable.csv")
    votes = summary.set_index("FileName")["MultiModalVote"]
    clips["perceived_vote"] = clips["clip_id"].map(votes)
    clips["perceived_emotion"] = clips["perceived_vote"].map(VOTE_TO_EMOTION)  # NaN for ties "X:Y"

    tabulated = pd.read_csv(raw / "processedResults" / "tabulatedVotes.csv")
    audiovisual = tabulated[tabulated.iloc[:, 0] >= AUDIOVISUAL_ROWS].set_index("fileName")
    clips["perceived_intensity"] = clips["clip_id"].map(audiovisual["meanEmoResp"]) / 100
    clips["perceived_raters"] = clips["clip_id"].map(audiovisual["numResponses"])

    demo = pd.read_csv(raw / "VideoDemographics.csv").set_index("ActorID")
    clips["age"] = clips["actor_id"].map(demo["Age"])
    clips["sex"] = clips["actor_id"].map(demo["Sex"])

    clips["has_video"] = [(raw / "VideoFlash" / f"{c}.flv").exists() for c in clips["clip_id"]]
    return clips


def check(clips: pd.DataFrame) -> None:
    """Invariants every downstream step relies on."""
    assert clips["clip_id"].is_unique
    assert clips["displayed_emotion"].isin(EMOTIONS).all()
    assert clips["perceived_vote"].notna().all(), "clip without ratings"
    assert clips["perceived_intensity"].between(0, 1).all()
    assert clips[["age", "sex"]].notna().all().all(), "clip without demographics"
    ieo = clips["sentence"] == "IEO"
    assert clips.loc[~ieo, "displayed_level"].eq("XX").all(), "intensity level outside sentence IEO"


def summary(clips: pd.DataFrame) -> None:
    n = len(clips)
    print(f"{n} clips, {clips['actor_id'].nunique()} actors, {clips['sentence'].nunique()} sentences, "
          f"{(~clips['has_video']).sum()} without video")
    print(f"displayed intensity defined on {clips['displayed_intensity'].notna().sum()} clips (IEO, LO/MD/HI)")
    print(f"perceived emotion tied on {clips['perceived_emotion'].isna().sum()} clips (masked)")
    for column in ("displayed_emotion", "displayed_valence", "perceived_emotion", "sex"):
        shares = clips[column].value_counts(normalize=True).round(3).to_dict()
        print(f"{column}: {shares}")
    print(f"perceived intensity: mean {clips['perceived_intensity'].mean():.3f}, "
          f"sd {clips['perceived_intensity'].std():.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", action="store_true", help="print label counts and shares")
    args = parser.parse_args()
    paths = load_paths()
    check_version(paths.raw)
    clips = build_index(paths.raw)
    check(clips)
    paths.processed.mkdir(parents=True, exist_ok=True)
    clips.to_csv(paths.processed / "clips.csv", index=False)
    print(f"wrote {paths.processed / 'clips.csv'}")
    if args.summary:
        summary(clips)


if __name__ == "__main__":
    main()
