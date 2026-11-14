# hunt5 report (Opus agent, 14 Nov night)

No big jump found. Best candidate: `submissions/v11_blockseg.csv`, est. +0.006 to +0.012 over v10 (0.4894).

## Candidates (all on top of v10)
| Rank | File | Change | Rows vs v10 | Expected LB |
|---|---|---|---|---|
| 1 | v11_blockseg.csv | Oct fraud device block (DV0473632–0474551) re-typed by sub-block position (monotone Viterbi over train class order); collusion kept | 107 | +0.006…+0.012 |
| 2 | v11c_simzone.csv | only the sim zone (54 rows ATO→sim) | 54 | +0.003…+0.006 |
| 3 | v11b_v7_blockseg.csv | same on v7-blend base | 307 | ≈ v11 + 0.002, unverified |

Reproduce: `.venv/bin/python src/make_blockseg.py --base submissions/v10_v6_octrules.csv --out submissions/v11_blockseg.csv`
(`--only sim_swap` → v11c; `--base submissions/v10b_v7_octrules.csv` → v11b). Needs `/tmp/P_bag4_R1.npy` from `bag_val.py`.

## Findings
1. Sim↔ATO "30% swap noise" was wrong: in-block type noise 1–3% (+12% unreported legit); most type errors are model errors.
2. Typing headroom: perfect typing of v6-flagged rows → Sep 0.459 → 0.539 (cashback↔phishing +0.021, ATO↔sim +0.019). A dedicated type model is worse (0.839 vs 0.855 type acc).
3. Block sim rows have ATO's profile; only rows/device + position separate them. Segmentation lifts Sep block class acc 0.48 → 0.95.
4. Generated fraud is already fully detected in Sep and Oct (~100 generated rows missed).
5. ip_risk_score / app_version dead in Oct even on label-free marker rows (IP AUC 0.77 → 0.54).
6. Oct legit second-phone IDs 100% dense.

## Negatives
Injected (legit-profile) rows: no signal on merchant popularity (AUC ~0.6, diffuse), pair repeats, degrees (0.49–0.53), money flow, shared-phone owner, partial duplicates, timestamps, ID ranges, month-long fan-in.

## Open question
Our test model matches all LB anchors (v10 est 0.496 vs 0.489; mule 0.503 vs 0.49; velocity 0.326 vs 0.34) and caps perfect generated-fraud typing at ~0.55–0.60. Leaders at 0.63 must have a tell on injected rows we haven't found, or test composition differs. Per-class probes of v10 (ATO, sim, phishing, cashback) would locate weak classes.
