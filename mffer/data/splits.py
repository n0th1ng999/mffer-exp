"""Actor-grouped, sex-stratified train/val/test splits.

Splits are created once and frozen as JSON files; every centralised, local-only and federated run reads
the same files. Re-running this module never overwrites an existing split.
"""
import argparse
import json

import numpy as np
import pandas as pd

from mffer.config import load_paths

FRACTIONS = {"train": 0.7, "val": 0.1, "test": 0.2}
N_SPLITS = 3


def make_split(actors: pd.DataFrame, seed: int) -> dict[str, list[int]]:
    """Assign each actor to train/val/test, separately within each sex so all parts keep the sex ratio.

    `actors` has one row per actor with columns `actor_id` and `sex`.
    """
    rng = np.random.default_rng(seed)
    parts: dict[str, list[int]] = {name: [] for name in FRACTIONS}
    for _, group in actors.groupby("sex"):
        ids = rng.permutation(group["actor_id"].to_numpy())
        n_val = round(len(ids) * FRACTIONS["val"])
        n_test = round(len(ids) * FRACTIONS["test"])
        parts["val"] += ids[:n_val].tolist()
        parts["test"] += ids[n_val:n_val + n_test].tolist()
        parts["train"] += ids[n_val + n_test:].tolist()
    assigned = sum(parts.values(), [])
    assert len(assigned) == len(set(assigned)) == len(actors), "an actor is missing or in two parts"
    return {name: sorted(int(a) for a in ids) for name, ids in parts.items()}


def split_file(split_id: int, dataset: str = "cremad"):
    """V1 splits in splits/, V2 (CREMA-D + RAVDESS, mffer.data.v2) in splits/v2/."""
    folder = load_paths().splits / ("v2" if dataset == "v2" else "")
    return folder / f"split_{split_id}.json"


def load_split(split_id: int, dataset: str = "cremad") -> dict:
    split = json.loads(split_file(split_id, dataset).read_text())
    parts = [set(split[p]) for p in FRACTIONS]
    assert all(not (a & b) for i, a in enumerate(parts) for b in parts[i + 1:]), "an actor is in two parts"
    return split


def rows(clips: pd.DataFrame, split_id: int, part: str, dataset: str = "cremad") -> np.ndarray:
    """Row indices (into clips.csv / cremad.h5) of the clips whose actor is in `part` of the split."""
    return np.flatnonzero(clips["actor_id"].isin(load_split(split_id, dataset)[part]).to_numpy())


def partition(clips: pd.DataFrame, train_rows: np.ndarray, n_clients: int, mode: str, seed: int) -> list[np.ndarray]:
    """Divide training rows among federated clients.

    mode "iid": clips are shuffled and dealt round-robin, so clients hold similar data.
    mode "actor": whole actors are dealt round-robin within each sex, so clients hold different people.
    mode "corpus": one client per corpus (V2: CREMA-D, RAVDESS), the natural non-IID partition (T3a);
    `n_clients` must equal the number of corpora.
    """
    rng = np.random.default_rng(seed)
    if mode == "iid":
        shuffled = rng.permutation(train_rows)
        return [np.sort(shuffled[i::n_clients]) for i in range(n_clients)]
    if mode == "actor":
        actors = clips.iloc[train_rows][["actor_id", "sex"]].drop_duplicates()
        owner: dict[int, int] = {}
        for _, group in actors.groupby("sex"):
            for i, actor in enumerate(rng.permutation(group["actor_id"].to_numpy())):
                owner[int(actor)] = i % n_clients
        client = clips.iloc[train_rows]["actor_id"].map(owner).to_numpy()
        return [train_rows[client == i] for i in range(n_clients)]
    if mode == "corpus":
        corpus = clips.iloc[train_rows]["corpus"].to_numpy()
        names = sorted(set(corpus))
        assert len(names) == n_clients, f"partition corpus: {len(names)} corpora {names}, {n_clients} clients"
        return [train_rows[corpus == name] for name in names]
    raise ValueError(f"unknown partition mode: {mode}")


def client_rows(clips: pd.DataFrame, split_id: int, n_clients: int, mode: str,
                dataset: str = "cremad") -> list[np.ndarray]:
    """The training rows of each client of `split_id`. The one place a client's data is defined, so a
    local-only client (L1) holds exactly the rows of the same federated client (F2): seed = split id."""
    return partition(clips, rows(clips, split_id, "train", dataset), n_clients, mode, seed=split_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    paths = load_paths()
    clips = pd.read_csv(paths.processed / "clips.csv")
    actors = clips[["actor_id", "sex"]].drop_duplicates()
    assert actors["actor_id"].is_unique, "an actor has more than one sex value"
    paths.splits.mkdir(parents=True, exist_ok=True)
    for split_id in range(N_SPLITS):
        file = paths.splits / f"split_{split_id}.json"
        if file.exists():
            print(f"{file.name}: exists, kept (splits are frozen)")
            continue
        split = make_split(actors, seed=split_id)
        file.write_text(json.dumps({"split_id": split_id, "seed": split_id, **split}, indent=1))
        sizes = ", ".join(f"{k} {len(v)}" for k, v in split.items())
        print(f"{file.name}: {sizes} actors")


if __name__ == "__main__":
    main()
