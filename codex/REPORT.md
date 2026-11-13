# Independent research review — 13 November 2026

Scope: read-only review of the competition pipeline, with all new artifacts under this directory. No leaderboard uploads, test labels, hand-labeling, or modifications to shared caches/source/docs. Training uses six threads. The experiment ledger below substitutes for updating `docs/experiments.md`, as explicitly required by the review's write restrictions.

## Executive assessment

There is no measured evidence here of a single missed feature explaining the leaders' approximately 0.08 advantage. The strongest concern is **prematurely treating low AUC as proof of irreducible label noise**. The current features provide weak discrimination on isolated structural fraud; that does not establish how those rows were generated. Likewise, similarity of app-version marginals is not proof of generator origin. Retain this as a working hypothesis, not a ceiling on attainable performance.

The 51% density adjustment has strong empirical support. A full-density experiment and targeted cleaning tests are recorded below. New neighbor-signature features have no persuasive improvement after held-out calibration.

## Findings ranked by expected practical impact

### 1. There is substantial error headroom even among observable connected fraud

**Measured diagnostic, NOT a deployable score or a forecast.** With the baseline's fixed tuned weights, 2,383 September fraud targets have another same-class labeled transaction connected within six hours **inside the sampled frame itself**. The model calls 490 of these legit and assigns 237 the wrong fraud type. Replacing only those connected fraud predictions with their true labels raises macro-F1 from **0.451322 to 0.522600**, keeping all other predictions—including false positives—unchanged.

Using connectivity from full-density labeled train instead gives 3,105 connected targets and an oracle score of 0.567239. Thinning removes visible same-class connectivity from 722/3,105 (23.25%) of those targets. This quantifies both recoverable error concentration and the cost of missing context. It does not prove the connected rows are all learnable; connectivity is defined using labels and may include accidental shared merchants.

**Recommendation / hypothesis:** prioritize event-level detection and type resolution on observable neighborhoods before concluding the gap requires learning completely ordinary isolated rows. Class-aware connected-component or pair/event models, and short multi-hop amount-flow features, merit a controlled ablation. For sampled connected fraud, current recall is 0.613 SIM swap, 0.692 mule, 0.739 velocity, 0.396 QR, 0.468 refund, and 0.183 phishing. The latter connections may be incidental; inspect aggregate event signatures, not individual test labels. The raw neighbor-signature ablation below did not realize this oracle headroom.

Evidence: `headroom.py`, `headroom.log`. Only train labels are used for this diagnostic; no test predictions are manually changed.

### 2. Fix model-selection calibration before ranking small improvements

**Measured.** `common.tune_weights` selects weights using the same September labels used to report F1. A deterministic two-way random split of September, tuning on one half and predicting the other, gives:

| Model | Argmax | Same-set tuning, two passes | Cross-fitted tuning |
|---|---:|---:|---:|
| Existing seed-0 no-IP baseline | 0.440990 | 0.451322 | 0.446656 |
| Existing four-seed average | 0.442003 | 0.455717 | 0.449347 |
| New raw neighbor signatures | 0.440112 | 0.451843 | 0.446441 |

Thus bagging's measured gain is +0.002691 under this calibration test, versus +0.004395 with same-set two-pass tuning. The supplied 0.4566 bag result uses a different calibration setup/pass count; this review does not overwrite it. The signature result's +0.000521 disappears under cross-fitting (−0.000215).

This is one split, not a confidence interval. Half-size calibration itself has variance, and connected events can occur in both halves. A stronger follow-up uses blocked/grouped calibration or a frozen first-half calibrator and later-half evaluation. **Do not spend a scarce upload on a +0.001 same-set tuning gain alone.**

Evidence: `calibration.py`, `calibration.log`. Uses the existing coordinate-ascent function and an equivalent confusion-matrix macro-F1 implementation. New training models use the same July–August/September targets, seed, hyperparameters, and cleaning as the baseline unless specified.

### 3. Density matching is supported; density and target coverage are different questions

**Measured, independent check:**

| Frame | Rows | Distinct senders | Rows with a repeated sender | Non-own-device share |
|---|---:|---:|---:|---:|
| Full September | 509,996 | 215,256 | 79.4691% | 8.1458% |
| September sampled at 0.51, seed 0 | 260,139 | 147,901 | 64.7093% | 8.1272% |
| October test, excluding November spillover | 260,536 | 148,182 | 64.6402% | 8.1217% |

This supports matching generic account activity density, but does **not** prove independent thinning within each fraud class. Changed QR mismatch prevalence (0.1922% sampled September vs 0.1455% October) already demonstrates that matching density does not eliminate fraud mix/signature drift.

**Hypothesis / next experiment:** training targets need not be restricted to one 51% realization forever. Existing multi-seed training recovers some target coverage. A controlled next comparison could combine complementary sampled training frames while holding September fixed, then compare against an equal-compute bag. Keep context density matched; do not double-count labels without accounting for weights.

**Controlled model experiment:** built features on all 1,824,642 train+test rows, then retained exactly the same seed-0 fitting targets, masks, labels, weights, and September targets as baseline. Only the context density changed.

| Training context / validation context | Argmax | Tuned F1 | Delta vs matched baseline |
|---|---:|---:|---:|
| 51% / 51% baseline | 0.440990 | 0.451322 | — |
| Full / 51% | 0.389708 | **0.413249** | **−0.038073** |
| Full / full | 0.461553 | 0.470701 | +0.019379, density-mismatched diagnostic |

The tempting full/full gain fails to transfer to the sparse validation frame. **Keep the density correction.** This experiment holds target count fixed; it does not test training on every target, count normalization, or mixed-density augmentation. Evidence: `density.py`, `density.log`, `full_density_same_targets.parquet`. No labels are used to construct context features.

### 4. Do not assume six-hour isolation means fake-cashback noise

The supplied experiments already show isolated cashback has AUC around 0.61 and multi-day partners. Yet `iso7` drops it using a six-hour same-label connectivity criterion. The `iso6.py` ablation keeps cashback and cleans only mule, velocity, layering, SIM swap, takeover, and collusion. This is a direct check of an inconsistent cleaning assumption, not a proposal to discard all cleaning.

For confident learning or further cleaning, use predictions generated out of fold by time/event groups, evaluate on unchanged September labels, and compare per-class recall. Never redefine validation labels based on model confidence. Relabeling and the iso6 experiment are recorded below.

### 5. Current neighbor-signature expansion is a negative result

**Measured.** Added 56 label-free features: counts of neighbors with SIM age <24h, device age <3d, PIN retries, amount/balance >0.8, login <30s, non-own devices, tiny payments, and COLLECT requests. Neighborhoods: sender/receiver/device ±1h, sender incoming preceding hour, receiver outgoing following hour, sender/receiver ±7d. Features exclude self where appropriate, use the same sampled frame, and never use neighbor labels.

Same-set tuned F1 0.451843 versus 0.451322 baseline; cross-fitted F1 0.446441 versus 0.446656 baseline. This particular expansion is not an upload candidate. It does not rule out event-level graph classification, multi-hop flow conservation, amount-decay paths, or a different learner. Those remain **untested hypotheses**, not claimed gains.

### 6. Reproducibility bugs and small leakage should be repaired

**Measured audit / inspected code:**

- The present seed-0 cache aligns exactly with `sample_frame(..., seed=0)`: 1,058,023 rows, 159 columns including metadata. Current `/tmp/y_val.npy` matches the cache labels. Train/test each have unique transaction IDs and share zero IDs. There is no demonstrated prediction misalignment for the current seed-0 baseline.
- `exp.py` writes a single global `/tmp/y_val.npy`, and probability arrays contain no IDs. `train_full.py` reads this shared label file while using seed-0 validation IDs. Length assertions cannot detect an equal-length shuffled array. `make_sub.py` gets labels from the seed-0 cache and avoids that particular shared-label bug, but still cannot verify probability order. Save probabilities with IDs, class order, source fingerprint, and validation label provenance.
- `/tmp/feat_s<seed>.parquet` has no feature-code fingerprint. Changes to `features.py` can silently coexist with stale caches across seeds. Key caches by version/source hash; verify columns and semantics before bagging.
- `exp.py --drop` accepts prefix wildcards; `train_full.py --drop` only matches exact names. An ablation such as `--drop s2_*` would train different validation and final models. Current exact-name IP removal is unaffected.
- `exp.py` supports `iso7relabel` and `isoall`, while `train_full.py` implements only `iso7` and silently accepts other strings. A successful alternative cleaning experiment requires implementing the same policy in final training, not merely passing its flag.
- `data/conn.parquet` is built using all labeled months. July–August cleaning can inspect September labels within six hours of the boundary. A deliberately permissive bidirectional join found **at most seven** July–August fraud rows connected across that boundary. This is a real but tiny leakage route; it cannot plausibly explain 0.08 macro-F1. Build cleaning flags using the fitting partition only. The proxy-validation metric is label-selected and is not a clean-test performance estimate.
- `r_nround_senders_7d` counts a sentinel `-1` as one distinct sender whenever any non-round row is present. It therefore includes an extra non-round bucket, rather than counting only round-amount senders. This is a feature semantics bug; the model can partially accommodate it, so no gain is claimed from fixing it.
- Both model-training entry points hard-code 10 threads. Review scripts use 6, as requested.
- `cat_mismatch` is identically zero among merchant transactions in both train and test. Dropping it is justified; it is not an overlooked signal.

Evidence: `audit.py`, `audit.log`, inspected source snapshots and `input_hashes.txt`. No source repair was made because the reviewer was instructed to write only here.

## Reproduction and artifacts

Run from the project root with `PYTHONDONTWRITEBYTECODE=1` to avoid writing source bytecode. Numba cache output is redirected under this directory. Do not run the model commands concurrently; another agent shares this machine.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python codex/audit.py > codex/audit.log 2>&1
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python codex/research.py > codex/research.log 2>&1
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python codex/iso6.py > codex/iso6.log 2>&1
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python codex/density.py > codex/density.log 2>&1
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python codex/calibration.py > codex/calibration.log 2>&1
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python codex/history.py > codex/history.log 2>&1
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python codex/headroom.py > codex/headroom.log 2>&1
```

Source hashes and the reviewed feature/common source snapshots are saved here. Scripts import `src` at runtime, so reproduce against those recorded versions. Validation IDs and labels are saved beside the new probability arrays. `results.jsonl` is an append-only metric ledger; re-running appends duplicate experiment entries. No public LB gain has been measured for any new experiment.

## Class-specific entity-history check

**Measured negative for a simple positive-risk rule.** Fit class-specific sender/receiver/device counts on full July–August; divide each class count by total entity count + 10, then take the maximum of the three entity rates for September. Evaluate isolated rows of that class versus legit, using the supplied full-train connectivity flags. AUCs: mule 0.4528, velocity 0.4560, layering 0.4612, SIM 0.4578, takeover 0.4795, phishing 0.5292, QR 0.4574, refund 0.4817, cashback 0.4867, loan 0.5059, collusion 0.4499.

These do not support “past same-class fraud means higher risk.” Below-chance AUC is an inverse association, not no information: it may reflect recurring merchant traffic, entity selection, or fraud rotation. This does not rule out a nonlinear, properly time-cross-fitted history feature. Prior “any fraud history hit rate” alone was insufficient to dismiss all target encoding. Evidence: `history.py`, `history.log`.

## Cleaning ablations completed

The iso6 hypothesis did not improve overall September F1: 0.449690 versus 0.451322 baseline (−0.001631). Cashback's own F1 rose only from 0.274914 to 0.277873, with losses elsewhere. `iso7relabel` yielded 0.449613 (−0.001709); cross-fitted calibration yielded 0.445212 versus 0.446656 baseline. Keep the existing dropping policy on the evidence available. These are single-seed comparisons, not estimates of the average over sampling seeds.

The relabel experiment changes training labels only; original September labels remain untouched. Relabeled rows enter the same 25% legit subsampling with weight 4. The isolated cashback test retains 633 additional fraud targets (141,887 fitting rows versus 141,254 baseline), with all other settings held constant.

## Blending check

Prespecified equal-weight blends were tested; no blend-weight search was performed.

| Validation probabilities | Same-set tuned F1 | Cross-fitted F1 |
|---|---:|---:|
| Four-seed existing bag | 0.455717 | 0.449347 |
| Baseline + signature model, 50/50 | 0.453072 | 0.446421 |
| Four-seed bag + signature model, 50/50 | **0.457108** | **0.450537** |

The last blend is the only new candidate with a positive gain against the existing bag in both measurements: +0.001392 same-set and +0.001190 cross-fitted. This is modest and based on one calibration split, not statistical significance or a measured leaderboard improvement. The base/signature blend's apparent gain disappears on cross-fitting. The iso6 cross-fitted result is 0.443112, reinforcing its negative result.

Reproduce the blend arrays before calibration:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python codex/blend.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python codex/calibration.py > codex/calibration.log 2>&1
```

`bag_signature_blend_P.npy` contains **validation** predictions, not a submission. To test on the leaderboard, first fit the signature-feature model on all provided training months with the same sampling/cleaning policy, align test IDs, and blend 50/50 with the existing bag. Recheck on another calibration split or sampling seed before committing an upload. There is no evidence this alone closes the leaders' gap.

## Numeric experiment ledger

All entries below use unchanged seed-0 September labels and two-pass weight tuning; no leaderboard score is available. Deltas are against the existing seed-0 baseline, not the stronger bag. Full/full is a deliberately mismatched diagnostic.

| Experiment | Argmax F1 | Tuned F1 | Tuned delta |
|---|---:|---:|---:|
| signature | 0.440112 | 0.451843 | +0.000521 |
| relabel | 0.436555 | 0.449613 | -0.001709 |
| baseline | 0.440990 | 0.451322 | +0.000000 |
| iso6 | 0.440088 | 0.449690 | -0.001631 |
| fulltrain_sampledval | 0.389708 | 0.413249 | -0.038073 |
| fulltrain_fullval_optimistic | 0.461553 | 0.470701 | +0.019379 |


## Recommended next actions

1. Preserve the current density-matched, iso7-cleaned baseline and existing bag. Neither relabeling nor retaining isolated cashback won here.
2. Replicate the bag/signature blend on another calibration split or sampling seed. Its measured gain is small; avoid optimizing many weights against the same September labels.
3. Spend the main research budget on event-level models and multi-hop amount-flow features. Use the connected-error diagnostic to measure which class/confusion improves, not just aggregate F1. Simple raw neighbor signatures alone were insufficient.
4. Add prediction IDs/class-order manifests, feature-cache fingerprints, and consistent train/validation flag semantics before parallel experimentation expands. Restrict label-cleaning connectivity to the training fold.
5. Keep “isolated rows are generated legit” and “all entity histories are useless” explicitly provisional. The evidence supports weak present signals, not an information-theoretic ceiling.

Final verification: every review script parsed successfully; SHA-256 checks confirmed all six audited source files, the seed-0 cache, and baseline probabilities still match the initial recorded hashes. All new outputs remain in this directory. No upload was performed.
