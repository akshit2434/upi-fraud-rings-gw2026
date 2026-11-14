# hunt2: structure in October (test) that Jul–Sep doesn't have

Method: features from `/tmp/feat_v4_s0.parquet` joined to raw rows (`load.py`). I compared Oct test rows (260,582) against 51%-thinned Sep (260,139)
and used thinned Aug (268,430) as a month-to-month noise control. I also joined v6_R1_bag labels and probabilities.
All scripts are in /tmp/hunt2. No changes to src/, docs/ or submissions/.

## 1. The finding: QR payments to P2P receivers with a blank `qr_vpa_mismatch` (140 rows, v6 calls all of them legit)
- `novel.py` scanned every 1/2/3-way combo of raw categoricals (txn_type, receiver_type, merchant_category, mismatch status, app, pin,
  new_payee, own device, away, os, kyc, hour, ...). Train has **0** rows of `txn_type=QR & receiver_type=P2P & qr_vpa_mismatch=NaN`
  across 2,199 QR→P2P rows in Jul–Sep, and the null share was 0.0 in every month. Test has **140** such rows.
  It is the only sizeable test-only combo. The few others are night bursts that v6 already flags.
- Train QR→P2P overall is 80% qr_swap, 13% legit, 3% phishing, 2% collusion. These rows carry a shop category
  (grocery 61, food 42, fuel 26, pharmacy 11), which is the textbook qr_swap: "thinks they're paying the shop".
- It is **substitution, not new volume**. QR→P2P per 10k rows: Jul 14.5, Aug 14.2, Sep 13.5, Oct 12.2 (all rows) but only 6.9 if the null rows are left out.
  Checked QR→P2P fell 352→179 (Sep51→Oct) and the 140 null rows fill the gap. So the mismatch check was blanked on about 44% of QR-swap payments:
  "rings adapt", which matches the known drop in mismatch=1 (19.6→14.6 per 10k).
- Receiver history (`qr3.py`): train rows *to* the 139 null-row receivers are 39.5% fraud, and 44 of those receivers took qr_swap payments in train.
  For checked QR→P2P test receivers the figures are 39.1% fraud and 55 qr_swap. For 140 random P2P test receivers they are 7.9% fraud and 1 qr_swap. Same receiver pool as the known qr_swap.
- Why v6 misses them: v6 gives them P(fraud) median 0.010 (max about 0.026). In train, the mismatch column being filled when it "shouldn't be", or blank when it
  should be, is a strong generator artefact. 15.8% of all train fraud sits in anomalous mismatch patterns, e.g. PAY→MERCHANT with mismatch=0 is 88% fraud,
  QR→MERCHANT with blank is 89% fraud (`nulls2.py`). The model never saw QR→P2P with a blank.
- Candidate file: `/tmp/hunt2/cand_v6_qrnull.csv` = v6 with those 140 rows set to qr_swap (`make_cand.py`). Nothing else changes.
  Rough impact: with test qr_swap ≈ 800–1,200 rows, v6 at 338 predicted and about 0.8 precision, and the rule at about 0.8–0.87 precision,
  qr_swap F1 rises by about +0.11 to +0.13, so **macro about +0.009 to +0.011**. That clears the +0.005 bar. It can't be validated on Sep because the pattern doesn't exist there,
  so only the LB can test it (about 42 of the rows fall in public).

## 2. Binned-cell scan Oct vs Sep51 (`cells.py` → `cells.parquet`, 22.6k cells over 21 binary/binned fields)
- Single-field marginals are almost exactly stationary: Oct/Sep ratio is 0.99–1.03 for txn_type, category, hour, new_payee, sim/device freshness, away, pin and login.
- Cells over-represented in Oct are almost all **event cells v6 already flags at about 100%**:
  - P2P fan-in ≥3 senders/h: 191→358 rows (v6 100%)
  - onward money within 1h: 350→506
  - night sender bursts: 14→89
  This fits "fraud events whole in test".
- `amt_just_under` +1,124 rows (×1.09): **negative**. It tracks the festival amount shift exactly (<5k band ×1.13 overall vs ×1.14 for just-under).
- Every other low-v6-flag excess cell has |z| ≤ 4.3 among 22k cells (multiple-testing noise). The top one is QR&P2P&mismatch≠1, which is the null pattern above.
- Adversarial Sep-vs-Oct LGB on **model-legit rows only**, with time-drift columns removed (merchant/account ages, merchant traffic): AUC **0.534** (`adv2.py`).
  - Top features: amt_over_bal, device/sim age (festival spending).
  - 125 of the 140 null-QR-P2P rows land in its top 1%.
  - The rest of the top-1% excess (+664 rows) is diffuse PAY/QR with amt/bal 0.3–0.9. That is amount drift with no coherent cluster: **negative**.

## 3. Reverse direction: where did velocity bots go?
- Device-level view, non-own phones (`vel3.py`): "many-sender" bot devices (≥3 senders, <2 rows/sender, ecom/gaming, login ≤10s)
  fell from 14.0 to 6.5 rows per 10k (15 devices → 4). Few-sender fast bots are flat (2.8→2.6).
  No compensating excess anywhere among non-own ecom/gaming rows: single-device rows 72.8→70.8, fast-login rows 23.7→16.2. **Velocity bots disappeared rather than relocated.**
  Only 4–15 bot devices per month exist in a 51% sample, so this is consistent with the event-level sampling Codex found.
- The bursts that remain changed shape. In Oct they are a few senders hammering one device, including at night:
  - night: 40 rows from 3 senders on 1 device
  - day: 60 rows, 6 senders, 2 devices
  In train, connected velocity was 872 rows from 601 senders on 53 devices. v6 still flags them all as velocity_bot.
- Night sender-bursts with fresh SIM on own phone and amount ≈0.9×balance rose from 12 to 45 in Oct. v6 calls them sim_swap, which matches the profile.
- The velocity probe (F1 ≈ .34 with 263 predicted) implies about 1,100 true velocity rows. Train velocity is 62% own-device and only 32% in device bursts,
  so the missing ones are the legit-looking "injected" kind. **No Oct-specific velocity group was found.**

## Negatives (honest)
- No large legit-predicted excess cluster besides the 140 null-QR rows. The ~2.7% unflagged fraud is not a structurally new October pattern
  visible in these columns. It is consistent with the flat injected-noise rows.
- The just-under excess and the adversarial excess are festival amount drift.
- There is no new category/app/kyc/hour combination (novel scan, ≥8 test rows, ≥4× expected), apart from the QR blank and bursts already flagged.
