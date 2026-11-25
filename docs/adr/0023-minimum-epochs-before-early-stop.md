# 0023. Protocol 3: no early stop before 20 epochs (or rounds), patience 10, budget 150

- **Status:** Accepted, 24 Nov 2026
- **Where:** `mffer/train.py` (`PROTOCOL = 3`, `DEFAULTS["min_epochs"]`, `DEFAULTS["patience"]`,
  `DEFAULTS["max_epochs"]`, `should_stop`, used by `EpochCheckpoint`), `mffer/federated.py` (`RoundLog.finished`)
- **Amends:** [0004](0004-training-budget-90.md) (patience 5, budget 90)

## Context

On 24 Nov the queue ran C2, D1–D4, F1, F2, L1, F3 and F4 under protocol 2 (patience 5 on val valence
macro-F1, ADR 0014). For the first epochs the valence head answers "negative" for every clip (~68% of
clips), so val valence macro-F1 stays exactly at the majority value 0.2705 while the emotion heads are
still learning (F1 split 0: val displayed-emotion macro-F1 0.13 → 0.27 over rounds 0–5). Patience 5 then
stops the run at epoch 6 with best epoch 1:

- every F1, F2, L1, F3 and F4 run (each round or client sees half the data, so learning is slower);
- U2 (video only) on all three splits;
- centralised runs sometimes: D3 stopped at 12, C2 splits 0 and 2 at 12 and 15 epochs.

Those runs measure the stopping rule, not the model or federated learning. The reason for this change is
the flat monitor seen on validation and training logs, not any test score.

## Decision

- Early stopping cannot trigger before **20** completed epochs (rounds): `min_epochs: 20`.
- Patience rises from 5 to **10**. A single macro-F1 over 3 classes on 19 validation actors is noisy;
  10 epochs is about 1/9 of the 90-epoch budget.
- The budget rises from 90 to **150** epochs (rounds). The first protocol-3 run (F2 split 0, 24 Nov) ran all
  90 rounds with its best round at 84, still improving: with 20 epochs of minimum and patience 10 the cap,
  not early stopping, was ending it. Early stopping still ends most runs well before 150; `max_epochs` only
  decides when to stop, so a run that stopped early is identical under either budget.
- Unchanged: monitor val displayed-valence macro-F1 (0014), the best epoch over **all**
  epochs is kept (also one before epoch 20), test scored once with it.
- One function, `train.should_stop(done, since_best, config)`, decides for centralised and federated runs.
- `PROTOCOL = 3`: every finished trained run of protocol ≤ 2 is "invalid (old protocol)" (0017). B0 stays.

## Consequences

- All 24 Nov paper runs (U1, U2, C1, C2, D1–D4, F1–F4, L1) are invalid and are run again; the dev sweep
  (0017) runs under protocol 3 too.
- A run costs at least 20 epochs and at most 150 (about 3.3 h for M3 on the RTX 4080): ~26 min for M3 on the RTX 4080 (~1.3 min per epoch with validation).
- Whether the valence head then leaves the majority class in FL is an empirical question; the balanced
  valence loss (0019) addresses the collapse itself and is tested in the dev sweep.
