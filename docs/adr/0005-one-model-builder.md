# 0005. One model builder; every configuration is a preset of flags

- **Status:** Accepted, 18 Nov 2026
- **Where:** `mffer/models.py` (`build_model`, `PRESETS`, `spec_from_config`), `configs/runs/*.yaml`

## Context

The design ladder (M1 → D1 → … → M3) must attribute a score difference to **one** design change. Separate
model files drift apart in details nobody meant to vary (initialisation, head definitions, losses).

## Decision

A single `build_model(spec)` builds every model from declared flags: visual encoder (simple or deep),
gating, text, demographics and loss weights. The fusion type follows from them (hierarchical when text or
demographics are on). A configuration is a preset, and neighbouring ladder steps differ by one flag.
Compilation is a separate function, `compile_model`, so federated clients can recompile.

## Consequences

- A diff between two configs is the complete list of what changed.
- A new architecture idea becomes a new flag in the builder, not a new model file.
