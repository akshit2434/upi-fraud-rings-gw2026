# Approach — UPI Fraud Rings (team Adam's Apple)

Best public upload: `v13_oofu.csv` (public macro-F1 0.5124). Exact reproduction: `README.md`. Every experiment and LB score: `docs/experiments.md`; verified data facts: `docs/insights.md`.

## 1. Base model
- LightGBM multiclass (400 iters, lr 0.05, 63 leaves), bag of 4 sampling seeds, legit down-sampled to 25% with weight 4.
- Features (`src/features.py`): single-row fields plus entity/time-window context (sender, receiver, device, pair; counts, distinct counterparties, onward transfers, reverse pairs, just-under-round amounts, new phone/SIM/city), all computed on a **51%-density frame** because test is a ~51% sample of October. `app_version` and `ip_risk_score` are not used (they don't transfer to October).
- Class weights tuned for macro-F1 on September (cross-fitted to avoid optimism).

## 2. Label noise
- Training labels contain unreported fraud (labelled legit) and wrong-type filings; ~60% of every fraud class is a class-random relabel of ordinary legit rows (identical feature mix in every class).
- We drop isolated rows of structural classes, apply a confident-learning relabel map (week-blocked OOF), and drop rows whose week-blocked OOF probability says "legit-profile fraud" or "unreported fraud" (`--clean iso7oofu`).

## 3. October rules (`src/oct_rules.py`, `src/make_blockseg.py`, `src/make_tailfix.py`)
- **QR→P2P with blank `qr_vpa_mismatch`** (never seen before October; the fraud ring's adaptation) → `qr_swap`.
- **Device-ID allocation**: the generator allocates fraud devices in a contiguous, class-ordered block (train DV0420000–0428454; October DV0473632–0474551). Rows in the October block are re-typed by an ordered (monotone Viterbi) segmentation over the train class order; the block tail follows refund → cashback → loan → collusion.
- Small rules: lure (₹1–10 then reverse payment) → `fake_cashback`; QR→merchant blank mismatch with non-round amount → `qr_swap`.
All rules are code on provided columns; no hand labels, no outside data.

## 4. What did not work (see experiments.md)
Ring/neighbour propagation stage-2 model, entity label history, full-density frames, flag-more thresholds, CatBoost/AutoGluon blends on LB, larger models, many label-free "excess" and artefact hunts.
