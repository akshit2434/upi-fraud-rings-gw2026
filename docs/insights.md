# Insights — UPI Fraud Rings (ML Gradient Wars 2026)

Facts verified from the data or official docs. Each line says how it was established.
Official problem statement: `data/upi-fraud-rings/PROBLEM_STATEMENT.md`. Event info: `hackathon_info.md`.

## Task & scoring (official)
- 12-class per-transaction label; macro-F1 over all 12. All-`legit` scores ~0.08.
- Public LB = 30% of test, final = other 70% scored on each team's **best public upload**.
- Organizers: "0.40 is excellent, we don't expect 0.50". Public LB top was 0.547 at ~11:00 IST on 13 Nov.
- 10 uploads/day (reset midnight IST). Code + approach must be submitted before 15 Nov 10:00 IST; winners must reproduce.
- Using `test.csv` for connection features is explicitly allowed.

## Data facts (measured)
- train 1,564,060 rows (Jul 1 – Oct 10; 147 rows dated Oct, mostly refund_abuse); test 260,582 (Oct 1 – Nov 9; 46 rows in Nov).
- ~450 MB in pandas; loads in <1 s. Mac (M4, 16 GB) is plenty. No GPU needed.
- Class shares: legit 96.3%; mule 0.66, velocity 0.52, qr_swap 0.46, layering 0.44, phishing 0.43, sim_swap 0.26,
  ATO 0.26, refund 0.21, collusion 0.16, cashback 0.16, loan 0.13 (%).
- Device ID `DV0xxxxxx` matches the account number for the account's own phone (92% of rows); others are "new phones".

## Test is a ~51% random sample of October (measured)
- Test 8.4k rows/day vs train ~17k/day. Subsampling Sep at p=0.51 reproduces test's rows/sender (1.76) and distinct senders (148k).
- => Every window/count feature must be computed on a **51%-density** frame for training too (`sample_frame`).
  Different sampling seeds = free bagging.

## Drift / traps (measured)
- `app_version`: train only 3.x/4.x, test only 5.0–5.2. Useless — never use.
- `ip_risk_score`: test distribution shifted (median 0.18→0.21) and compressed (p99 0.58→0.47). In train, real fraud ~0.70 rank vs legit 0.50.
  In the 147 Oct-dated train rows the relation looks inverted (tiny sample). Suspected trap — currently used as within-month rank
  (`ip_rank_month`). **Needs an LB A/B (with vs without).**
- Amounts ~15–20% higher in Oct (festival). Added `amt_rank_month`.
- `qr_vpa_mismatch=1` rate drops 19.6 → 14.6 per 10k in Oct (rings adapt?). Null rate stable at 64%.
- Adversarial val (Sep vs Oct) AUC 0.93, driven by **edge effects of my own features** (unbounded "time to next event",
  all-time counts). Fixed in features v2: long windows past-only, time-to-event capped at 1 day.

## Label noise structure (measured, important)
- Only 12–48% of each fraud class is "connected" (shares sender/receiver/device with a same-class row within 6h, or chain link):
  loan 12%, phishing 15%, qr 16%, refund 20%, cashback 26%, collusion 29%, ATO 39%, sim 41%, mule 42%, layering 46%, velocity 48%.
- **Isolated fraud rows mostly look exactly like legit** (same sim age, device age, ip rank ≈0.50, categories). E.g. isolated
  sim_swap has SIM age 9,140h (= legit). These are either injected label noise or deliberate camouflage — unlearnable.
  Exceptions that keep single-row signal: qr_swap (mismatch 0.35), phishing (COLLECT 31%), refund (REFUND 29%), loan (lending 27%).
- Connected fraud has clear signatures: velocity = new phone, ecommerce/gaming, ~29s login→pay, many accounts on one device;
  sim_swap = fresh SIM, amt ≈1.5× balance, night; ATO = away from home, new phone, PIN retries; layering = huge amounts, chain;
  collusion/refund/qr = round amounts.
- Unreported fraud: inside clear bursts (e.g. velocity burst on DV0421188) a few rows are labelled legit → neighbour labels can fix them.
- Consequence: validation (noisy Sep labels) underestimates test (clean labels). Model has high precision (~0.7), low recall (~0.3).
  Class-weight calibration should be checked on the public LB.

## Pitfall
- pandas 3 parses datetimes at non-ns resolution: `astype('int64')//1e9` is WRONG. Use `(ts - T0).dt.total_seconds()`.

## Connected vs isolated fraud — model view (measured on Sep val, model v2@400)
- Recall on connected fraud: mule .60, velocity .70, layering .79, sim .63, ATO .56, collusion .56, loan .79, cashback .36, qr .43, refund .44, phishing .31.
- Recall on isolated fraud: ~0 for mule/velocity/layering/sim/ATO/collusion (median own-class prob 0.001–0.004);
  ~0.27–0.33 for phishing/qr/refund (real single-row fraud), loan .16, cashback .06.
- F1 on Sep with isolated fraud rows dropped ("proxy val") = 0.62–0.65 vs 0.44 on full noisy Sep. LB top 0.547 sits between.
- Hypothesis (unverified until LB): isolated struct-class rows in train are injected noise; clean test has fewer of them.
- Stacking with neighbour-probability aggregates gave no gain on the noisy val (exp #5).

## LB calibration (measured 13 Nov 12:05 uploads)
- Public LB ≈ full noisy Sep val + ~0.005–0.013. Proxy val (isolated fraud removed) overestimates massively → test still contains
  the "ordinary-looking" fraud. Primary metric = full Sep val.
- Dropping isolated rows of the 7 struct classes from training helps on LB (0.447 → 0.459).
- No synthetic consistency leak: account age, SIM age, device age are independently sampled per row (exp #11).

## Feature drift into October (measured, exp #15)
- `ip_rank_month` loses its fraud signal in October (fraud-like rows 0.53 vs 0.50, was 0.69 vs 0.50 in Sep) → likely THE trap column. Drop it.
- Stable: new_payee, pin_retries, away_from_home, amt_rank_month, login_to_txn_sec. Weaker: device_age_days.
- Isolated refund/loan/collusion/cashback rows usually have a same-class partner within 7 days (multi-day incidents);
  isolated mule/velocity/sim/ATO/phishing/qr/layering mostly don't (exp #13).

## Why we are capped (measured, exps #21–26)
- Isolated rows of mule/velocity/layering/sim/ATO/collusion: AUC 0.54–0.56 vs legit with all features; entity label history
  same as legit; old-app-version share 16–18% (= legit 15%, connected fraud 51%). They are legit-generated rows → unlearnable.
- Isolated phishing/qr/refund/loan/cashback are real (AUC 0.61–0.73, v3 share 27–33%).
- `app_version` is the strongest trap: 3.x fraud rate 7.6% vs 4.x 2.9% in train; test is 100% 5.x.
- Device-ID blocks (DV0420000–428500) are generator fraud blocks in train — not transferable to Oct.
- Oct differs in class mix: fewer velocity bursts (half), fewer QR mismatches; signatures otherwise stable Jul→Sep.
- Starter notebook = 0.285 on Sep. Leaders prepped overnight (data out 00:40 IST).

## LB results so far (13 Nov 17:30)
- v4_f3_noip 0.4663 > v4_f3_ip 0.4644 > v3_iso7_full 0.4594 > v3_iso7_proxy 0.4542 > v2_lgb400 0.4470.
- Multi-day features: +0.005–0.007 on LB. Dropping IP: +0.002 (consistent with Oct drift check, within noise).
- LB − Sep-val gap is stable at +0.013–0.015 → Sep val remains a good proxy for model selection.

## Label noise, refined (exps #31–35)
- Wrong-type noise is real and large between look-alike classes: ~30% of sim_swap/ATO labels swapped; mule↔layering, velocity→mule.
  Mistyped rows carry the other class's textbook profile, so on clean test labels our model is likely RIGHT on them.
- Connected fraud is almost fully detected (19/1,816 missed in Sep sample); remaining val errors there are type-noise.
- Sep val is biased against denoising (its labels have the same noise); LB rewards denoising more (iso7: +0.012 LB vs +0.004 val).
- Same-set weight tuning inflates small gains; use cross-fitted calibration (tune on one random half, score the other).
- Fake cashback vs phishing confusion is probably a sampling artefact (₹1–10 lure row missing in the 51% sample), not label noise.

## Test fraud prevalence (measured via all-legit probe, 13 Nov 18:43)
- Public test fraud share ≈ 4.1% (train labels 3.66%). Not a hidden-unreported-fraud scenario; we predict 1.39%.
- Our LB 0.4685 ⇒ mean fraud-class F1 ≈ 0.42; leaders 0.547 ⇒ ≈ 0.51. Gap is spread across classes or concentrated in Oct-drifted classes.
- Relabel R1 + 4-seed bag: +0.0022 on LB (x-fit val predicted +0.0036).

## Night of 13→14 Nov (exps #40–51)
- Single-split x-fit has ±0.003 noise; use the 5-split mean (`src/blend_eval.py`) before believing any gain < 0.005.
- Model family / ensembling (LGB bag, CatBoost, AutoGluon-XGB), graph-event features and event smoothing are all ≤ +0.0023 → the
  current feature set is saturated; LB 0.50 needs new signal, not modelling.
- Oracle bound: perfect on full-density-connected fraud alone = 0.488 Sep. Isolated struct-class fraud (52–71% of those classes) is
  temporally uniform, id-free, entity-history-free and has legit's app-version mix → unlearnable from the provided columns.
- Isolated phishing/qr/refund/loan/cashback carry fraud-like profiles (old-app 23–33%, just-under amounts 1999/4999/9999) → remaining headroom.

## 14 Nov afternoon (exps #52–57)
- LB probes: mule test F1 ≈ .49 (Sep .40), velocity ≈ .34 (Sep .48). Recall can't be inferred without precision.
- Test sampling: legit traffic is row-thinned (~51%), but **fraud events appear whole** (about half of events kept intact). Event-preserving
  training frames did not help (−0.0015). v6 already flags nearly every visible October event row.
- Isolated struct fraud is injected at a flat 1.31–1.32% of rows every month, has legit's feature/app-version profile, and is very likely
  present (as fraud) in test. No column separates it (combined AUC .566).
- EM prior shift Sep→Oct: velocity ×0.43, mule ×1.42, refund ×1.6, cashback ×2.35, loan ×4.1 — but bimodal probs mean re-weighting moves ~0 rows.

## Device-ID allocation artefact (exp #66) — corrects the 'device blocks don't transfer' note above
- Device IDs are allocated sequentially per period: own phones (= account number), then a FRAUD phone block, then legit second
  phones (each used once). Train block DV0420000–0428454 is 88% fraud with sub-blocks in fixed class order
  (velocity, mule, velocity, layering, sim, ATO, phishing, qr, refund, cashback, loan, collusion). October's block is
  DV0473632–0474551 (564 test rows). Rows there that look legit are the 'injected' fraud; the device number is the only tell.
- October adaptation: QR→P2P with blank qr_vpa_mismatch (140 rows, never in train) are qr_swap — LB +0.0098.
