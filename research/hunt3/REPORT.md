# hunt3: more October dodges and adaptations, on top of v9 (public 0.4783)

Setup: the frame is `/tmp/hunt2/frame.parquet`, which holds v4 features on the 51%-thinned seed-0 frame (Sep thinned to 260k rows, Oct is the full test).
I joined v9 labels to it (`base.py`). For Sep I used v6's bagged validation predictions (`/tmp/P_bag4_R1.npy` × `preds/v6_R1_bag_weights.npy`, from `sepp.parquet`),
so "Sep pred" means a model trained on Jul–Aug.
Impact estimates come from `impact.py`. It is a per-class F1 calculator that assumes base precision 0.8 and true per-class test counts T estimated from marker scaling (see §3).
Calibration check: it gives +0.0090 to +0.0105 for the known QR rule (v6→v9) at flip precision 0.8 to 0.9. The real LB gain was +0.0098.

## Ranked findings and candidate CSVs (all in /tmp/hunt3, each changes only its targeted rows of v9)

| rank | rule | rows changed | evidence for precision | est. LB gain | file |
|---|---|---|---|---|---|
| 1 | **A DEVBLOCK**: October fraud-device ID block | 212 legit→fraud (+6 fraud→legit) | Sep analogue: 90% of model-legit block rows are labelled fraud even with noisy labels; class assignment is 82% right | **+0.009 … +0.016** (central +0.013) | `cand_v9_A.csv` |
| 2 | B QRM_BLANK: QR→MERCHANT, blank mismatch, amount not a multiple of 100 | 21 legit→qr_swap | 0 such rows in train with paise; profile identical to the 140 QR→P2P blank rows | +0.001 | `cand_v9_B.csv` |
| 3 | C LURE: cashback lure events (tiny ₹≤10 P2P then reverse payment ≤7d) | 7 (legit/phishing→fake_cashback) | train: lure events with gap >6h are 102 cashback vs 0 phishing | +0.0003…0.0005 | `cand_v9_C.csv` |
| 4 | D QR_NOCAT: QR→P2P without category | 3 legit→phishing | train 79% phishing (60/76) | +0.0001 | `cand_v9_D.csv` |
| – | **all together** | 248 | | **+0.0105 … +0.018** (central ≈ +0.015) | **`cand_v9_ABCD.csv`** |

Build: `.venv/bin/python /tmp/hunt3/rules3.py --rules ABCD --out /tmp/hunt3/cand_v9_ABCD.csv` (default `--base` = `submissions/v9_v6_qrrule.csv`; it works on any base).
The rules are deterministic, written in code and use no labels from test. Suggested upload: `cand_v9_ABCD.csv`, or `cand_v9_A.csv` alone if you want a clean A/B of the device-block hypothesis.

### 1. Device-ID allocation leaks the fraud class, and October has its own new fraud block (the big one)
The generator hands out device IDs in order (`dev.py`, `dev2.py`, `dev4.py`):
- `DV0000000–0419999`: each account's own phone.
- `DV0420000–0428454`: **fraud devices**, 87–93% fraud-labelled. They sit in contiguous sub-blocks by class, in this fixed order:
  velocity bots (multi-row), mule, velocity (1-row), layering, sim, ATO, phishing, qr, refund, cashback, loan, collusion.
  The ~12% "legit" rows here have fraud's profile (marker 30%, old-app 54%), so they look like unreported fraud.
- `DV0428455–0473631`: legit second phones. Every ID is used exactly once (a contiguous run) and the fraud rate is 0.07%.

October repeats the layout with fresh IDs. **`DV0473632–0474551` is the October fraud block** (564 rows).
The legit second-phone run starts at 474552 (contiguous from there, 7,462 rows, v9 fraud 6).
v9's fraud calls in the block follow the same class order: mule → layering → ATO (devage ≈1 d) → phishing → qr → refund/collusion → loan/collusion.
v9 calls **212 block rows legit**. Their profile (QR 40%, paise 68%, marker 0%, devage 13 d) is the same as the 192 Sep block rows that v6 called legit, and **90% of those carry fraud labels** in Sep (`devprof.py`, `devrule.py`).
These rows are the legit-looking "isolated/injected" fraud. On a non-own phone it gets a device from its class's block, and that is the only column that gives it away.
- How the class is assigned: order-aware nearest neighbours (`order_rule.py`, K=3).
  Take the majority class of the 3 nearest v9-fraud rows on each side by device number. If the two sides differ and the train order has exactly one class between them, use that class.
  This recovers the 1-row velocity sub-block between mule and layering (Oct 473750–473850, QR-heavy), which plain KNN mislabels.
  On the Sep analogue the class is right for **82%** of truly-fraud rows (plain KNN K=5: 72%; the model's best fraud class: 13.5%).
- The flips go to: mule 60, velocity 30, phishing 29, qr 24, ATO 23, layering 14, refund 12, loan 9, collusion 5, cashback 4, sim 2.
- A3: the 6 v9-fraud rows on the legit second-phone run (5 ATO, 1 qr) are set to legit.
- A2 (rows on old train-block devices that v9 calls legit) changed 0 rows: the 6 such rows are on devices that have no fraud labels in train.
- Negative: re-classing rows that v6/v9 already call fraud by block neighbours lowers class accuracy on Sep (0.892 → 0.875, `reclass.py`). So the rule keeps the model's classes.
- Caveat: this exploits a synthetic-ID artefact, not fraud behaviour. It is allowed (a coded rule, no hand-labelling), but say so in the write-up.
  It also contradicts insights #22 ("not transferable to Oct"): the block moved, it did not vanish.
  The same idea could become a training feature: "device in the fraud block" plus the neighbours' classes.

### 2. QR→MERCHANT with blank mismatch and paise amounts (21 rows)
In train, QR→MERCHANT with a blank mismatch (678 rows) is 89% fraud. **Every** one of those amounts is a multiple of 100, except 40 integer-amount collusion rows (`qrm.py`, `qrm2.py`).
October has 21 blank rows that are not multiples of 100, and v9 calls all 21 legit. Their profile matches the 140 QR→P2P blank rows exactly (`qrm3.py`):

| group | paise share | shop categories | ecommerce |
|---|---|---|---|
| these 21 rows | 0.81 | grocery 43, food 33, pharmacy 19, fuel 5 % | 0 |
| the 140 QR→P2P blank rows | 0.81 | same | 0 |
| legit QR→MERCHANT | 0.99 | – | 22% |

So this is the same "dodge the check" adaptation, applied to qr_swap whose payee is a merchant ID. Estimated precision 0.6–0.85. About 6 of the rows should fall in the public split.

### 3. A field-level fraud marker, used to estimate October class sizes (diagnostic)
- `qr_vpa_mismatch` set on a **non-QR** row is a generator fraud fingerprint (`marker.py`, `mrate.py`).
  Train: 88% fraud. About 31–35% of connected fraud carries it, 10–16% of isolated single-row-class fraud, 1–3% of isolated struct-class fraud, and 0.09% of legit.
- Sep51 1,236 such rows → Oct 1,134. v9 calls only 8 of the October ones legit.
- Scaling each class's marked count by its Sep marked/true ratio gives estimated October true counts:
  mule ≈2.3k, qr ≈1.4k, phishing ≈870, layering ≈860, cashback ≈800, ATO ≈790, velocity ≈630, loan ≈570, sim ≈490, collusion ≈470, refund ≈430.
  The total of ≈9.6k = 3.7% matches the 4.1% public fraud share measured by the all-legit probe. These are the T values in `impact.py`.
- In-month PU model (`pu.py`, `pu_eval.py`): LGB trained on marked vs unmarked non-QR rows inside October, out-of-fold.
  Its high-scoring unmarked rows are already v9-flagged (≥97%). **There is no October-specific marked-lookalike cluster that v9 misses.** Negative.

### 4. Per-class marker battery, Sep51 → Oct (`battery.py` → `battery.csv`)
Every class marker is stable (ratio 0.9–1.1) or rises because whole fraud events are kept in test.
For the rising ones, v9 flags at least Sep's true rate: fan-in ≥2/1h 458→692 (v9 86%), onward ≤1h 350→506, >20k onward 111→249, reverse-pair payments 53→86.
Deficits:
- velocity device bursts 556→269. Velocity bots really halved: the marked velocity rows also halved (133→66 ecommerce, `mkm.py`, `vel.py`, `vel2.py`). They did not move to another category or channel.
- QR mismatch=1 496→375. This is the known QR dodge.
- sender ≥2 lending/7d 142→111.

There is no sim/ATO dodge (`drain.py`): drains by SIM age, device age, away and PIN have the same fraud rates as Sep. No phishing amount dodge either (`phish.py`): just-under COLLECT 266→283, flagged.

### 5. Known fraudster receivers (`recv.py`, `pool.py`, `qrrecv.py`)
- P2P receivers with fraud history in train get 19–56% fraud on their later incoming payments.
- On Sep (history from Jul–Aug only), the model's legit calls on these receivers are only 2.7–10% fraud, close to the 2.4% residual everywhere. So the model already absorbs this signal.
- October incoming payments to these receivers are predicted at the same rates as in Sep. QR-swap receivers also receive phishing COLLECTs (shared ring pool), which v9 already flags.
- **No rule here clears the bar.**

### Other negatives (measured)
- Integrity checks: receiver prefix vs type, category vs merchants.csv, onboarding/opened dates, sender account age vs opened_on all have **0 violations in both train and test** (`consist.py`).
  `new_payee` doesn't match pair history (it is random), and account/merchant ID ranges carry no fraud blocks (`newp.py`, `acct.py`).
- Loan (`loan.py`, `loan2.py`): repeated evening lending victims are fully flagged (83/85). Isolated loan has legit's hour profile. There is no P2P loan-agent pattern.
- Mule slow fan-in or late onward (`mule.py`): no excess. Receivers with ≥3 senders/day are 100% flagged.
- Cashback vs phishing lures (`cash2.py`, `lure.py`): phishing-labelled lure events are a **distinct** pattern, not label noise. Their gap is always ≤6h, versus 29% >6h for cashback.
  So rule C only touches gap >6h and legit members.
- Integer vs paise amounts (`paise.py`): a strong fraud fingerprint on merchant payments, but the model already uses it (amt_is_int) and it has no October shift.
