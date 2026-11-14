# Research review 2 — 14 November 2026

No source or shared documentation edits, leaderboard uploads, or manual test labels. All new experiments/scripts are here. Each model uses 10 threads; model fits run sequentially. New experiment ledger is this report, overriding the usual docs-update habit per this task's explicit restrictions.

## Ranked findings

### 1. Independent 51% thinning is a poor model of October's visible fraud events

The prior density evidence establishes ordinary account-activity density, not within-event sampling. A new label-free diagnostic groups consecutive transactions sharing a phone with gaps ≤10 minutes, retaining groups with ≥3 senders and ≥10 rows. July–September contain 84 such events, all 20–60 rows (mean 39.77). October's seven events have sizes **29, 31, 32, 37, 40, 42, 58**, mean **38.43**. Full September mean is **42.64**, while independently thinned September is **21.28**.

Using the empirical training event-size distribution as the pre-thinning distribution, conditioning on observed size ≥10, the likelihood is maximized at retention **p=1.00**, versus the assumed p=.51. Log-likelihoods: −25.235 versus −71.815; relative likelihood of .51 is **5.89e−21**. Even for an original 60-row event, P(observing ≥58 rows at p=.51)=**4.82e−15**. These are model-conditional diagnostics, not a posterior probability or proof of the generator: October could instead have newly enlarged events.

Corroboration from independently defined cohorts:

| Label-free cohort | Full September events / rows | Thinned September events / rows | October events / rows |
|---|---:|---:|---:|
| Large shared-phone bursts | 25 / 1,066 | 25 / 532 | 7 / 269 |
| P2P receiver fan-in: consecutive gaps ≤1h, ≥3 senders | 151 / 742 | 36 / 170 | 76 / 410 |
| Sender bursts: gaps ≤30m, ≥2 rows, median SIM<48h or device age<7d | 316 / 868 | 116 / 260 | 168 / 444 |

Sender-burst mean sizes are 2.747 full September, 2.241 thinned September, and 2.643 October. Across these cohorts, **fewer largely intact events** is a better working hypothesis than uniformly missing half of each event. This invalidates the *universal* thinning claim; it does not justify full-density context for ordinary traffic. Train context should preserve events while sampling ordinary traffic.

Evidence: `research2_diag.py/.log`, `research2_thinning.py/.log`, `research2_events.py/.log/.json`. Thinned diagnostic uses pandas sampling seed 0, distinct from the pipeline's NumPy seed-0 frame.

### 2. Visible event recovery alone cannot explain the entire leaderboard gap

The existing v6 submission already labels **263/269** October large-phone-event rows velocity_bot (six legit). It labels all **410/410** P2P fan-in rows fraud (383 mule, 27 layering), and **438/444** short suspicious sender-event rows fraud (238 SIM, 199 takeover, one mule).

These are predictions, not verified test labels. They show that broadcasting a fraud flag across these particular visible cohorts adds very few rows. Thus the density correction is a real framing issue but not demonstrated to recover the missing ~0.16 macro-F1. Alternative event definitions, multi-day episodes, and type assignment remain possible; no 0.6+ solution is established here.

The earlier report's broad merchant-connected oracle was already corrected in experiment 31. This review uses P2P-only receiver links and does not resurrect that misleading oracle.

### 3. The per-class probes do not identify recall or hidden class counts

With known class F1 F and prediction count M, F=2TP/(M+N). Without precision or the true class count N, neither recall nor N is identified. The statements “mule recall ~35%” and “velocity recall ~20%” require a high-precision assumption. A low bot F1 could mean missed bots, wrong typing of detected bursts, or both.

Even substituting a representative whole-test prediction count M=263 and F≈.34, feasible N ranges roughly from FM/(2−F)≈54 to M(2/F−1)≈1,284. These are illustrative whole-test extrapolations: the probe actually scores a public subset, whose prediction count is unavailable here. Public-score rounding and the legitimate-class F1 subtraction introduce further uncertainty. No new upload was made. Official local task/config documents say macro-F1 over 12 classes; the all-legit score is consistent with that. Server implementation was not independently audited.

### 4. Omitted raw features and removing label cleaning do not unlock a large gain

Added 52 features to the existing 153-feature baseline: timestamp seconds; fractional/last-digit numeric fields; sender/receiver city, OS, KYC and equality flags; device-city codes; within-month/type/category numeric ranks; distances/moduli around common payment limits. No app_version, IP risk, or raw entity-ID feature. July–August fitting and untouched seed-0 September labels; 400-iteration LGB with existing hyperparameters.

| Experiment | Mean x-fit F1, seeds 123/124/125 | Argmax F1 |
|---|---:|---:|
| Existing v4 seed-0 LGB, iso7 + R1 | 0.450811 | see existing run |
| Added raw features, iso7 + R1 | 0.451335 | 0.441329 |
| Added raw features, no cleaning or relabeling | 0.441475 | 0.435555 |

The +0.000524 feature gain is below the upload bar. Removing *both* cleaning and relabeling loses 0.009861; this joint ablation does not isolate their individual effects, nor prove clean-test behavior. Fitting targets: 141,254 cleaned versus 148,928 uncleaned. Runtime: 117s and 124s respectively. Three calibration splits are sensitivity checks, not independent dataset replications.

No duplicated transaction tuples (timestamp, sender, receiver, amount), nor duplicated five-field numeric fingerprints (SIM/device/account ages, balance, login time), were found across train+test. These specific copy/template shortcuts are negative; they do not establish impossibility of other signals.

Evidence: `research2_model.py augmented` and `augmented_unclean`; logs, validation probabilities, IDs, labels, feature importances; `research2_compare.py/.log`; `research2_results.jsonl`.

## Whole-event sampling experiment

`research2_event_sample.py` builds label-free components using same phone ≤30m, sender ≤1h, P2P receiver ≤1h, and receiver-to-next-sender money hops ≤1h. A seeded Bernoulli(.51) draw retains whole components; singletons are still independently sampled. Test context is untouched. Features use the existing builder; cache is isolated here. Frame: 797,867 retained training rows plus all test rows, largest component 60.

Both conventional row-sampled training and whole-event-sampled training are evaluated on **the identical whole-event-sampled September targets and context**, with unchanged observed labels. This is a controlled comparison under a newly motivated validation design, not verified clean-test performance. It uses current iso7/R1 training policy; the pre-existing R1 map's documented OOF temporal leakage limitation remains. No September labels are used to build graph components or context.

Measured results on identical 260,440 September targets:

| Training sampling | Fitting targets | Argmax F1 | Mean x-fit F1 (123/124/125) |
|---|---:|---:|---:|
| Independent rows | 141,254 | 0.449583 | **0.455728** |
| Whole components | 141,181 | 0.448323 | **0.454204** |

**Negative training result: −0.001524**, with split-wise deltas −0.004308, +0.002944, −0.003207. Building the event frame took 18.5s; the entire sequential two-model experiment took 234.7s. The current model scores ~0.0049 better on this changed validation set than on conventional validation, but this is a change of targets/context, not an improvement to the model. The validation redesign is motivated by measured event sizes; this implementation does **not** establish a useful model gain. It is one component definition and sampling seed, not a universal rejection of event-preserving training.

## Decision and artifact status

**No new submission candidate meets the +0.005 requirement.** No test probability/submission artifact was generated: the novel sampling approach lost to its controlled baseline, and the feature model gained only 0.000524. All new probability arrays are validation-only and named accordingly. No leaderboard attempt was used.

Ranked next action: replace the universal-thinning assumption with explicit background/event density checks, while retaining the existing production model on the present evidence. The observed event cohorts already have nearly complete predicted fraud coverage. Do not claim that the leaderboard gap has been explained, that isolated rows are mathematically unlearnable, or that probe F1 identifies recall. A clean-test improvement needs evidence beyond these noisy-validation comparisons.

Final checks: all six new Python scripts parse; no source changes appear in git diff. The pre-existing `.DS_Store` and `docs/experiments.md` modifications were left untouched. This task's numeric findings and negative results are recorded here and in the local result ledger.

## Reproduce

Run sequentially from the project root, using `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python`:

- `codex/research2_diag.py`
- `codex/research2_thinning.py`
- `codex/research2_events.py`
- `codex/research2_model.py augmented`
- `codex/research2_model.py augmented_unclean`
- `codex/research2_compare.py`
- `codex/research2_event_sample.py`

No model here establishes a score approaching 0.6. Do not interpret changed validation density or oracle/proxy scores as a measured leaderboard improvement.
