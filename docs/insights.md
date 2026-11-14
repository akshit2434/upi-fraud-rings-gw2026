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

## hunt5 (14 Nov night) — clean labels inside the fraud block, typing is the fixable loss
- Train fraud-block labels are ~clean: inside each class sub-block, other-fraud labels are 1–3% and legit 12% (unreported).
  So wrong-type noise is small; the sim↔ATO / velocity→mule / mule→layering 'swaps' seen on Sep are mostly MODEL errors
  (profile overlap), not label noise. Measured: labels vs device-run class (`research/hunt5/a14,a16,a18.py`).
- Block sim_swap rows look like ATO (new phone, old SIM); only rows/device (sim 43% single-row vs ATO 19.5%) and position separate them.
  v6 calls Sep block sim rows ATO (47/62). Ordered segmentation of the block (`src/block_seg.py`) types Sep block fraud rows at .95.
- Generated-fraud detection is complete: labelled fraud that v6 misses has legit's app-3.x share (.15) and IP (.20).
  IP and app version are dead in October even on label-free marker rows (AUC .54/.52 vs .77/.74 in train).
- Injected (legit-profile) rows: no signal on merchant popularity (weak, AUC ~.6 diffuse), pair repeats, money flow, degrees,
  shared-phone owner, partial twins. The only tell found remains the device block.
- October legit second-phone IDs are 100% dense (each ID used once) → October legit rows were not thinned after ID allocation.
- Typing headroom (exp #81): oracle types on v6-flagged Sep rows → .539 (from .459); a dedicated type model can't get it (#82),
  row-level type noise ≲3.5% (#83). Outside the block the confusable pairs (cashback/phishing, ATO/sim) overlap in all features.
- Upload candidate from hunt5: `submissions/v11_blockseg.csv` (v10 + block segmentation, 107 rows), est. +0.006…+0.012.

## hunt6 (15 Nov 00:00–) — legit-profile fraud is a constant-rate relabel of the main legit path
- Sep51 LP rows (non-block, isolated, model P(fraud)<.05) vs model-legit legit: every raw column, accounts/merchants join,
  entity-activity, own-profile deviation, gap, P2P-graph feature has AUC .48–.52; combined GBM .568 with zero precision at the top.
  The legit P2P graph itself is random (no common neighbours, no reverse edges) — there is no community structure to break.
- LP rate is flat 2.3–2.4% of main-path rows regardless of merchant popularity; halves at mega-merchants (>5k rows) and is ~0 on
  legit second phones → those are separate legit generator paths that were never relabelled. No rule can exceed ~2.4% precision.
- October has no new row-level population among model-legit rows (adversarial AUC .507 once amount/balance festival drift is removed).
- txn_id is not a hash of a counter or a seeded RNG stream (chance-level hits) → no generation order.

## 15 Nov 00:30 — where the gap to the leaders is (main session)
- Test has ~10,630 fraud rows (4.08% × 260,582); v12 flags 4,003. ~6.6k fraud rows are unflagged (≈ the injected/legit-profile share).
- Week-blocked OOF (/tmp/oof_s0.npy, iso7-trained LGB): 59–75% of every fraud class has OOF P(legit) > 0.95. Strongly bimodal: the legit-profile part is the same size in every class, merchant_collusion highest (73%).
- Device type of Sep fraud: own phone 86–94% for all classes except ATO (70% own, 26% fraud block) and velocity (60% own, 36% block). Fraud device block ≠ "generated": generated mule/layering/etc. live on own phones.
- Oct velocity_bot calls are 100% on train-block devices reused in Oct (device farms: one device, many senders, integer amounts, login_to_txn 1–6 s, merchant PAY).
- Ceiling arithmetic: perfect detection + typing of only the flaggable ~4k rows → class recall ≈ .38, F1 ≈ .55, macro ≈ .59. Leaders (0.63) must catch some legit-profile fraud; the 0.54–0.55 pack is consistent with near-perfect generated-fraud typing.
- `new_payee` is drawn independently of the observed pair history (P(new_payee=1) is .23 whether or not the pair was seen before)
  → not a usable consistency check. Structural/null-pattern cells and exact-value collisions show nothing for LP rows (exp #98).
- Ceiling: perfect on all non-LP Sep fraud while calling LP rows legit = .585 Sep macro-F1. Test (4.1% fraud, 43.5 marker rows/10k
  vs train 47) has the same generated share as Sep, so LB .634 is above that ceiling. Either leaders recover LP rows in a way we have
  not found (no row/entity/graph/time/ID axis works), or they get typing/precision near-perfect and benefit from public-split noise. Unresolved hypothesis.

## hunt7 (15 Nov 00:20–00:50) — excess / distribution-shift view of legit-profile fraud (all negative)
- An excess detector (classifier two-sample test, Aug legit vs Sep unflagged) scores Sep's unflagged fraud at AUC .49. Month-to-month drift
  (receiver traffic growth, amount rank, a festival mega-merchant in Oct) dominates every C2ST (AUC .69–.84) (exp #98).
- Entities that carry legit-profile fraud get no more rows than their history predicts (AUC .46–.53, #99). Tiny linked structures
  (bursts, same-amount pairs, fan-in, hops) occur at the same rate in Oct-unflagged as in Sep51-unflagged (#100).
- Upper bound: a supervised x-fit model on Sep unflagged rows (all features + entity popularity) reaches AUC .505, top-100 precision .03
  (#102). Raw ID axes other than the device number carry nothing (#103). Residual fraud is noise with respect to every provided column,
  so the remaining gap must come from detecting and typing flaggable rows, not from legit-profile fraud.
- **The October fraud 'excess' is train under-reporting (exp #108).** Label-independent generator fingerprints (fraud device block,
  non-QR filled mismatch, v6 P≥.99) are 11.4–12.8% legit-labelled, flat by month → ~12% of all fraud is labelled legit in train.
  Clean share 3.67%/0.88–0.90 = 4.08–4.18% matches the probe's 4.07%. There is no hidden October population to find: tail, cell,
  near-miss, chain and fan scans of v12-unflagged Oct vs Sep find no excess ≥ ~20 rows (#104–107). On clean labels v12's flagged rows
  are ≈97% precise, so extra rows must come from typing, not from looser thresholds (ab_flag002 agrees).
- (15 Nov 02:05, exp #116) Sep type confusions between cashback/phishing and ATO/sim are profile-consistent with the PREDICTED class (lure ₹1–10 rows labelled phishing; COLLECT just-under-round rows labelled cashback; own-phone new-SIM rows labelled ATO). These are event-level wrong-type filings (noise); on the clean test labels profile-based typing should be right, so the Sep oracle-typing gap (+0.08) overstates the real headroom. Exception: fraud-block rows, where type follows ID position (exploited by v11 block segmentation).
- (15 Nov 01:55, exp #115) Legit-profile fraud rows have the identical receiver/txn-type mix in every class (MERCHANT .64, PAY .56, QR .37) = legit generator mix; e.g. LP mule rows pay merchants 64% of the time (impossible by definition). Class-random relabel; not ring-linked (receiver sharing at base rate).

## hunt8 (15 Nov 02:00–02:15) — legit-profile fraud is an i.i.d. relabel; no deterministic trace (exps #121–126)
- Raw CSV strings: one writer and one float format for every row (trailing zeros stripped); digit-level chi² LP vs legit all null. Measured on str-dtype CSV.
- `sender_account_age_days` = round(ts − (opened_on + u)) with a hidden per-account opening time u; all 1.83M rows (LP, GEN, legit, test) fit one u per sender. Balance, device age and SIM age have no cross-row chain (independent draws).
- LP rows are relabelled existing legit rows (sender row counts equal the row-weighted legit distribution; insertion would add +1 row).
- Selection is i.i.d.: Poisson per day, not tied to same-class GEN days, no sender/receiver/device clustering (obs/exp 1.005/.993), class drawn independently of the sender's other LP rows. Per-class LP vs legit AUC .48–.51 on every column.
- ⇒ No rule, model or artefact can find LP rows in train, so none can in test. The leaders' edge must come from generated rows (detection, typing, October adaptations) or from public-split noise.
- Fraud device blocks are sparse (train 28% of IDs used, Oct 45%), legit second-phone zones 100% dense; per-class fill ~.5 (ATO/qr lower, sim higher) is identical in train and October, so gaps are a generator property. Oct velocity farms reuse train-block IDs never seen in train. Gaps carry no rows → no labelling use (exp #127).
- (15 Nov 02:35, #129b/#130) Per-class test F1 of v13 from LB probes: qr .549, mule .531, layering .512, phishing .477, velocity .362. Velocity is low because October reuses only 7 of 81 train bot-farm devices (269 rows) while legit-profile velocity stays at the train rate; no missed generated velocity found (no hidden bursts, no new farm devices, no farm sub-block in the Oct block).
- (15 Nov 04:45, #134b) Mule precision probes: test has ≈2,040 true mule rows; v13 flags 867 at precision ≈.89 (≈.94 top half, ≈.84 bottom half) ⇒ recall ≈38%. Legit-profile fraud is in the clean test labels; the leaders (fraud-class avg F1 ≈ .60) would need ≈45% mule recall at similar precision.
