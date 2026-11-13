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
- Python: `.venv/bin/python` (3.12; pandas 3, lightgbm, catboost, networkx, optuna, numba, polars). System python has no ML libs.
- AutoGluon lives in a SEPARATE env `.venv-ag/bin/python` (it pins older pandas). Exchange data via parquet; never install it into `.venv`.
- Run from the project root. Logs → `logs/`, test probabilities → `preds/` (always with `_test_ids.csv` + `_manifest.json`), uploads → `submissions/`.
- Feature caches are version-keyed: `/tmp/feat_<FEATURE_VERSION>_s<seed>.parquet` via `features.load_cached(seed)`.
  Bump `FEATURE_VERSION` in `src/features.py` whenever feature semantics change. v3 caches = code at git commit 559e3a3 (submissions ≤ v6).
- Code is under git (local only). Commit before every submission; note the commit hash in `docs/experiments.md`.

## Code map
- `src/features.py` — `load`, `sample_frame` (51% subsample of train + full test), `build_features` (edge-safe),
  `load_cached`, `apply_drop` (same drop semantics everywhere; `DEFAULT_DROP` = cat_mismatch, ip_rank_month).
- `src/common.py` — macro-F1, `tune_weights`, **`xfit_f1` (primary metric: cross-fitted weights)**, `clean_bad`, `get_conn`, `apply_relabel`.
- `src/exp.py` — validation (Jul–Aug → Sep), reports argmax / same-set / x-fit; saves `/tmp/P_<tag>.npy` + `_ids.npy`.
  Cleaning uses `data/conn_until_2026-09-01.parquet` (Jul–Aug labels only).
- `src/train_full.py` — all training months, multi-seed → `preds/<name>_test_P.npy`. `src/make_sub.py` → CSV with val-tuned weights.
- `src/oof.py`, `src/relabel.py` — week-blocked OOF and confident-learning relabel maps (`data/relabel_R1.parquet`).
- `src/bag_val.py` — bagged validation probs; `src/connectivity.py [--until]` — label-structure flags.

## Rules to respect
- Train only on provided data; no hand-labelling test. Must reproduce best upload from code → keep seeds fixed,
  keep every submission's exact command in `docs/experiments.md`.
- Never use `app_version` (test-only versions). Treat `ip_risk_score` as suspect.
- Compute all context features on the 51%-density frame (test is a ~51% sample of October).
- pandas 3: use `.dt.total_seconds()`, not `astype('int64')//1e9`.
- Model selection uses x-fit Sep F1; same-set tuned F1 overstates small gains. Upload only if ≥ +0.005 x-fit over the current best.
- Uploads are scarce (10/day; final = best public upload). No probes or near-duplicates without the user's OK.
