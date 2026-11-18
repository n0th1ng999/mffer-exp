"""Paths and run configuration. A run is fully described by its YAML file."""
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    raw: Path
    processed: Path
    runs: Path

    @property
    def splits(self) -> Path:
        """The frozen actor splits are versioned with the code, not regenerated per machine."""
        return ROOT / "splits"


def load_paths(file: Path = ROOT / "configs" / "paths.yaml") -> Paths:
    data = yaml.safe_load(file.read_text())
    return Paths(**{key: (ROOT / value).resolve() for key, value in data.items()})


def load_run_config(file: Path) -> dict:
    return yaml.safe_load(Path(file).read_text())
