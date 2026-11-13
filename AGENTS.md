# AGENTS.md — ML Gradient Wars 2026 (UPI Fraud Rings)

Goal: win the 48h round (13 Nov 10:00 → 15 Nov 10:00 IST). Metric: macro-F1 over 12 labels; final = private 70%.

## Read first
- `docs/insights.md` — verified facts about data, drift, label noise. Don't re-derive; extend it.
- `docs/experiments.md` — every experiment with scores, plus next ideas.
- `data/upi-fraud-rings/PROBLEM_STATEMENT.md` — official task; `hackathon_info.md` — event rules.

## Habit (mandatory)
- After every experiment or new finding, append a row to `docs/experiments.md` and any new verified fact to `docs/insights.md`
  (say how it was measured). Record negative results too. Note every LB upload and its public score.
- Keep `docs/` honest: hypotheses are marked as such; don't write a claim you didn't measure.

## Environment
- Python: `.venv/bin/python` (3.12; pandas 3, lightgbm, numba, polars). System python has no ML libs.
- Run from the project root. Logs → `logs/`, test probabilities → `preds/`, uploads → `submissions/`.
- Feature caches: `/tmp/feat_s<seed>.parquet` (delete or `--rebuild` after changing `src/features.py`).

## Code map
- `src/features.py` — `load`, `sample_frame` (51% subsample of train + full test), `build_features` (edge-safe).
- `src/exp.py` — validation harness (Jul–Aug → Sep), F1 at checkpoints, flags for params/drops.
- `src/train_full.py` — train on all train, multi-seed, write submission with val-tuned class weights.
- `src/common.py` — macro-F1, per-class weight tuning.

## Rules to respect
- Train only on provided data; no hand-labelling test. Must reproduce best upload from code → keep seeds fixed,
  keep every submission's exact command in `docs/experiments.md`.
- Never use `app_version` (test-only versions). Treat `ip_risk_score` as suspect.
- Compute all context features on the 51%-density frame (test is a ~51% sample of October).
- pandas 3: use `.dt.total_seconds()`, not `astype('int64')//1e9`.
