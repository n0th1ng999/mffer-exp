# 0013. Lint with ruff, no auto-formatter; hooks outside the experiment env

- **Status:** Accepted, 19 Nov 2026
- **Where:** `pyproject.toml` (`[tool.ruff]`), `.pre-commit-config.yaml`

## Context

Lint catches real defects in research code: unused imports, silent `zip` truncation, shadowed names. An
auto-formatter would rewrite 20 of 22 files and undo the hand-aligned constant tables and comments.
`pre-commit` depends on a `virtualenv` release that needs `packaging >= 26.3`, while `flwr 1.43` needs
`packaging < 26`.

## Decision

- ruff with rules E4, E7, E9, F, W, I, B and UP at 120 columns, and no formatter.
- `zip()` always takes `strict=True`, so a length mismatch raises instead of truncating.
- pre-commit runs ruff and file hygiene on every commit, and the test suite on every push.
- `splits/` is excluded from the hooks (0002).
- pre-commit is installed with Homebrew, outside the `mffer` environment, so the experiment environment
  stays identical to `requirements-macos-arm64.lock`.

## Consequences

- Style is kept by hand. Lint stays strict.
- A new machine needs `brew install pre-commit && pre-commit install --hook-type pre-commit --hook-type pre-push`.
