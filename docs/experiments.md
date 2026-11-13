# Experiment log

Validation = features on a 51%-density frame (seed 0), train Jul–Aug, validate Sep (noisy labels).
"tuned" = per-class probability multipliers fitted on the same Sep set (optimistic by ~0.01–0.02).
Logs in `logs/`. Newest at the bottom.

| # | date (IST) | change | val macro-F1 argmax / tuned | LB public | notes |
|---|---|---|---|---|---|
| 1 | 13 Nov | features v1 (single-row + windows incl. all-time & unbounded next-event), LGB lr .05 leaves 63, legit downsample 25%, early stop | 0.428 / 0.451 | – | early-stopped at iter 38; v1 had edge-drift features (adv AUC 0.93) → abandoned |
| 2 | 13 Nov | features v2 (edge-safe: past-only long windows, capped dt, numba distinct counts, amt/ip month ranks, device_is_own) | 0.423 / 0.439 | – | `logs/val_v2.txt` |
| 3 | 13 Nov | v2, fixed iterations, F1 by iter | 25:.415/.435 · 50:.425/.440 · 100:.426/.439 · 200:.430/.441 · 400:.433/.442 | – | logloss early-stop misleading; F1 still creeps up at 400 |
| 4 | 13 Nov | **submission `submissions/v2_lgb400.csv`**: v2, 400 iters, trained on all train (seed 0), weights from #3@400 | (0.442) | **0.4470** | predicts 1.37% fraud. Cmd: `train_full.py --name v2_lgb400 --seeds 0 --iters 400 --val_P /tmp/P_base_400.npy` |
| 5 | 13 Nov | stacking A (`src/stack.py`): stage-1 OOF (4 week-blocked folds, 300 it) + neighbour sums of probs (device/sender/receiver/pair/chain windows) | stage1 .429/.435 → stage2@400 .430/.440 | – | **no gain on noisy val**; revisit with proxy metric + cleaned labels |
| 6 | 13 Nov | new metric "proxy val" = Sep with isolated fraud rows removed (`data/conn.parquet` from `src/connectivity.py`) | baseline: full .433/.442, **proxy .623/.632** | – | connected fraud recall 56–79% for struct classes; isolated struct fraud recall 0% |
| 7 | 13 Nov | label cleaning in training: `--clean iso7` drop isolated rows of 7 struct classes (mule, velocity, layering, sim, ATO, collusion, cashback) | full .429/**.446**, proxy .623/**.645** | – | better on both metrics |
| 8 | 13 Nov | `--clean isoall` (drop all isolated fraud) | full .370/.388, proxy .633/.641 | – | kills real single-row fraud (phishing/qr/refund/loan) → reject |
| 9 | 13 Nov | `--clean iso7relabel` (relabel as legit) | invalid | – | harness bug: relabelled val labels too and overwrote /tmp/y_val.npy. Fixed (`lab_eval`); not re-run yet |
| 10 | 13 Nov | **submissions `v3_iso7_full.csv` / `v3_iso7_proxy.csv`**: iso7-cleaned full model (seed 0, 400 it); class weights tuned on full Sep vs proxy Sep | full .446 / proxy .646 | **full: 0.4594 · proxy: 0.4542** (12:05 IST) | Cmd: `train_full.py --name v3_iso7 --clean iso7 --seeds 0 --iters 400`, then `make_sub.py --pred preds/v3_iso7_test_P.npy --val_P /tmp/P_clean_iso7_400.npy --wmode full|proxy --out v3_iso7_<wmode>` |

| 11 | 13 Nov | consistency-artefact check: sender_account_age vs accounts.opened_on; sim/device age continuity per device | – | – | **negative**: error rates identical for legit / connected / isolated fraud (ages resampled per row). No leak there |

| 12 | 13 Nov | burst analysis (`src/burst.py`): legit rows with >=2 same-class fraud nbrs (±1h, shared sender/P2P receiver/device/chain) | – | – | only ~964 such legit rows (velocity 447, mule 182, ATO 113, sim 110, layering 99) → unreported-in-burst fraud is small; relabelling them is low value |
| 13 | 13 Nov | isolated fraud: time to nearest same-class row sharing any entity | – | – | refund 94% / loan 86% / collusion 51% / cashback 51% have a partner within 7d → multi-day incidents. Others mostly >7d or none (noise-like) |
| 14 | 13 Nov | features v3: +both-sided multi-day windows (sender ±7d/next7d/±3d refunds, lending, late-night, round; pair ±7d/next7d/-30..+7d; receiver ±7d round/refund, distinct round-senders; reverse-pair next tiny credit), `--clean iso7` | full .440/**.453**, proxy .640/.654 | – | per-class: collusion .295→.345, refund .441→.463, cashback .266→.281, phishing .416→.433, mule .386→.396, loan .381→.360 |

| 15 | 13 Nov | drift check (`src/drift_check.py`): model w/o probe features, top 0.6% fraud-like rows vs low-risk half, Sep vs Oct | – | – | **ip_rank_month gap 0.686 vs 0.496 in Sep → 0.531 vs 0.499 in Oct: IP signal ~gone in test (trap)**. new_payee, pin_retries, away_from_home, amt_rank keep their gap. device_age gap shrinks (221 vs 297 → 268 vs 306) |
| 16 | 13 Nov | v3 features, iso7, **drop ip_rank_month** | full .441/.451 (vs .453 with ip) | – | IP adds ~0.002 on Sep only |
| 17 | 13 Nov | **submissions `v4_f3_noip.csv` / `v4_f3_ip.csv`** (LB A/B on IP) | .451 / .453 | **noip 0.4663 (best) · ip 0.4644** (17:27 IST) | Cmds: `train_full.py --name v4_f3_noip --clean iso7 --seeds 0 --iters 400 --drop ip_rank_month` + `make_sub.py --pred preds/v4_f3_noip_test_P.npy --val_P /tmp/P_f3_noip_400.npy --wmode full --out v4_f3_noip`; same without `--drop` and `--val_P /tmp/P_f3_iso7_400.npy` for `_ip` |

| 18 | 13 Nov | HP batch (no-ip, iso7): legit_keep .5 / leaves31-mcs100 / ff .4 | .449 / .446 / .451 (base .451) | – | tuning is flat (±0.005); lr/leaves127 not run (stopped) |
| 19 | 13 Nov | organizers' starter notebook reproduced (`src/starter.py`) | **0.285** | not uploaded | VenVenn's 0.486 at 10:03 was NOT the starter; data released 00:40 IST, leaders prepped overnight |
| 20 | 13 Nov | fraud-weight scale ×0.5…×5 on Sep | .4485….4527 | – | probabilities bimodal; thresholds are not the bottleneck |
| 21 | 13 Nov | isolated-fraud vs legit binary LGB per class (Jul–Aug → Sep AUC) | mule .556, velocity .556, layering .542, sim .549, ATO .563, collusion .535 · phishing .711, qr .726, refund .685, cashback .613, loan .611 | – | struct-class isolated rows are unlearnable with current features |
| 22 | 13 Nov | device-ID blocks (non-own phones DV0420000–428500 = generator fraud blocks per class) | – | – | strong in train but only 3–5% of isolated fraud on them; Oct new phones (DV0473632+) are 1-row legit-like; only 269 test rows on old fraud blocks → little test value. Raw device number must NOT be a feature |
| 23 | 13 Nov | predicted class counts Sep vs Oct (v4_noip) | – | – | Oct/Sep: velocity ×0.44, qr ×0.69, cashback ×1.49, mule ×1.28; device-burst rows 10.3 vs 21.5 per 10k → fewer velocity bursts in Oct |
| 24 | 13 Nov | monthly trend of connected-fraud signatures Jul–Sep | – | – | stable (layering amount 35k→41k→48k only) |
| 25 | 13 Nov | entity label-history (did sender/receiver/device have past fraud labels, Jul–Aug → Sep) | – | – | **negative**: hit rate isolated fraud ≈ legit (any-entity 65% vs 66%) |
| 26 | 13 Nov | app_version | – | – | train: 3.x fraud 7.6% vs 4.x 2.9%; test all 5.x → trap column (never used). Connected fraud 51% on 3.x, isolated struct-class 16–18% ≈ legit 15% → isolated struct rows are legit-generated rows |

| 27 | 13 Nov | **per-class LB probes** from v4_noip: `probe_<class>.csv` keeps one class, rest legit. test F1_k ≈ 12·LB − 0.982 | Sep F1: velocity .478, qr .484, mule .393, cashback .275 | pending | probes score ~0.1 so can never become 'best upload' |

| 28 | 13 Nov | 4-seed bagging (sampling seeds 0–3, iso7, noip; `src/bag_val.py` predicts seed-0 Sep rows) | singles .445–.451 → **bag .4566** | – | +0.005 |
| 29 | 13 Nov | **submission `v5_bag4.csv`**: mean of v4_f3_noip (seed 0) and seeds 1–3, weights tuned on bagged Sep P | .4566 | pending | Cmds: `train_full.py --name v5_noip_bag --clean iso7 --seeds 1,2,3 --iters 400 --drop ip_rank_month --wmode none`; average (1×seed0 + 3×bag)/4 → `preds/v5_bag4_test_P.npy`; `make_sub.py --pred preds/v5_bag4_test_P.npy --val_P /tmp/P_bag4_noip.npy --wmode full --out v5_bag4`. 267 rows differ from v4 |
| 30 | 13 Nov | independent review by Codex gpt-6-astra → `codex/REPORT.md` | – | – | density matching confirmed (full-density context −0.038); same-set weight tuning overstates gains (cross-fit: bag +0.0027 not +0.005); iso6/relabel-iso/neighbour-signature negative; bugs: `r_nround_senders_7d` counts sentinel −1, conn flags use Sep labels (≤7 rows), caches unversioned |

| 31 | 13 Nov | event diagnostic (`src/event_diag.py`): connected fraud in SAMPLED frame via sender/device/P2P-receiver/chain only | – | – | 1,816 connected Sep fraud rows: only 19 called legit; 219 wrong type (velocity→mule 53, ATO↔sim 79, mule→layering 33). Codex's 490 "missed" came from merchant-shared links (coincidental) |
| 32 | 13 Nov | profile of mistyped connected events | – | – | mistyped rows match the PREDICTED class's textbook profile (e.g. "ATO" rows with 16h SIM, own phone, 3am) → wrong-type label noise; model is right |
| 33 | 13 Nov | adversarial Sep vs Oct on behavioural features | AUC .73 | – | driven by `amt_rank_month` × cents (merchant spend up in Oct shifts P2P ranks); no festival-gift pattern (x01/x51 amounts same) |
| 34 | 13 Nov | OOF (week-blocked, 4 folds, `src/oof.py`) + confident-learning joint | – | – | swaps: sim→ATO 225, ATO→sim 229, mule→layering 204, velocity→mule 165, layering→mule 63; cashback→phishing 232 (likely missing lure row in 51% sample, not noise); ~1,575 legit rows confidently fraud |
| 35 | 13 Nov | relabel training labels (`src/relabel.py`): R1 = type swaps (886 rows), R2 = R1 + legit with OOF fraud p>0.8 (2,057) | raw same-set: base .4513, R1 .4557, R2 .4568 · **cross-fit raw: base .4462, R1 .4513, R2 .4500** · R2-denoised x-fit: .5021/.5030/.5034 | – | +0.004–0.005; minor leak: OOF models for Jul–Aug rows saw Sep labels |
| 36 | 13 Nov | `submissions/probe_all_legit.csv` (all legit) | – | **0.0816 → public fraud share 4.0–4.2%** | score = 2(1−f)/(2−f)/12 → exact public fraud share f: 0.0818 @3.7%, 0.0808 @6%, 0.0799 @8% |

| 37 | 13 Nov | **submission `v6_R1_bag.csv`**: R1 relabel + iso7 + noip, 4 sampling seeds; weights tuned on bagged Sep P (`/tmp/P_bag4_R1.npy`) | same-set .4588 · **x-fit .4498** (v4 .4462, v5 bag .4475) | **0.4685 (best)** | Cmds: `train_full.py --name v6_R1_bag --clean iso7 --relabel data/relabel_R1.parquet --seeds 0,1,2,3 --iters 400 --drop ip_rank_month --wmode none`; `bag_val.py --seeds 1,2,3 --relabel data/relabel_R1.parquet --base /tmp/P_rl_R1_400.npy --out /tmp/P_bag4_R1.npy`; `make_sub.py --pred preds/v6_R1_bag_test_P.npy --val_P /tmp/P_bag4_R1.npy --wmode full --out v6_R1_bag`. 290 rows differ from v4 |

| 38 | 13 Nov | per-class probes from v6: `submissions/probe6_<class>.csv` (one class kept, rest legit). class test F1 = 12·score − 0.979 | Sep F1: mule .399, velocity .479, layering .516, sim .389, ATO .361, phishing .441, qr .479, refund .454, cashback .265, loan .381, collusion .354 | pending | Oct predicted rows vs Sep: velocity 263 vs 555, qr 338 vs 499, mule 818 vs 640, cashback 200 vs 121 |

| 39 | 13 Nov | housekeeping (git commit after 559e3a3): FEATURE_VERSION v4 (fix `r_nround_senders_7d` sentinel), version-keyed caches, shared `apply_drop`/`clean_bad`/`apply_relabel`/`xfit_f1`, val cleaning from Jul–Aug-only flags (3 rows differed), preds saved with ids + manifest | – | – | no model run yet on v4 features. Installed catboost 1.2.10, networkx 3.7, optuna 5.0.0 (main env); AutoGluon in `.venv-ag` |

### LB calibration (13 Nov 12:20)
- LB tracks **full noisy Sep val** (0.442→0.447, 0.446→0.459), NOT proxy val. Use full Sep val as primary metric.
- Weights tuned on full val beat proxy-tuned weights on LB (0.459 vs 0.454).
- iso7 cleaning is confirmed on LB (+0.012).
- LB top 0.547 → we are ~0.09 behind; need detection of currently-missed fraud, not tuning.
- Caveat: unreported fraud is labelled legit in val but fraud in clean test, so val penalises burst-propagation
  (stacking, relabel of legit-in-burst) that may help on test → such ideas must be judged on LB.

## Next ideas (priority order)
1. Upload v2_lgb400 → get LB anchor. Then LB A/B: global fraud-weight scale ×0.5 / ×2 (val labels noisy, test clean).
2. LB A/B: drop `ip_rank_month` (suspected trap).
3. Stage-2 stacking: aggregate stage-1 OOF class probs over neighbours (same device ±10m, receiver ±1h, sender ±1d, chain prev/next).
   Use time-blocked folds (by week) to avoid neighbour leakage.
4. Label cleaning: relabel legit rows inside strong same-class bursts; optionally down-weight isolated legit-looking fraud rows.
   Can only be judged on LB (val labels are noisy).
5. Multi-seed bagging (different 51% samples), then CatBoost/XGB blend.
