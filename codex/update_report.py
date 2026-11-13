p='codex/REPORT.md'
s=open(p).read()
for i in range(5,0,-1):s=s.replace(f'### {i}.',f'### {i+1}.')
needle='## Findings ranked by expected practical impact\n'
s=s.replace(needle,needle+'''
### 1. There is substantial error headroom even among observable connected fraud

**Measured diagnostic, NOT a deployable score or a forecast.** With the baseline's fixed tuned weights, 2,383 September fraud targets have another same-class labeled transaction connected within six hours **inside the sampled frame itself**. The model calls 490 of these legit and assigns 237 the wrong fraud type. Replacing only those connected fraud predictions with their true labels raises macro-F1 from **0.451322 to 0.522600**, keeping all other predictions—including false positives—unchanged.

Using connectivity from full-density labeled train instead gives 3,105 connected targets and an oracle score of 0.567239. Thinning removes visible same-class connectivity from 722/3,105 (23.25%) of those targets. This quantifies both recoverable error concentration and the cost of missing context. It does not prove the connected rows are all learnable; connectivity is defined using labels and may include accidental shared merchants.

**Recommendation / hypothesis:** prioritize event-level detection and type resolution on observable neighborhoods before concluding the gap requires learning completely ordinary isolated rows. Class-aware connected-component or pair/event models, and short multi-hop amount-flow features, merit a controlled ablation. For sampled connected fraud, current recall is 0.613 SIM swap, 0.692 mule, 0.739 velocity, 0.396 QR, 0.468 refund, and 0.183 phishing. The latter connections may be incidental; inspect aggregate event signatures, not individual test labels. The raw neighbor-signature ablation below did not realize this oracle headroom.

Evidence: `headroom.py`, `headroom.log`. Only train labels are used for this diagnostic; no test predictions are manually changed.
''')
s=s.replace('PYTHONDONTWRITEBYTECODE=1 .venv/bin/python codex/history.py > codex/history.log 2>&1','PYTHONDONTWRITEBYTECODE=1 .venv/bin/python codex/history.py > codex/history.log 2>&1\nPYTHONDONTWRITEBYTECODE=1 .venv/bin/python codex/headroom.py > codex/headroom.log 2>&1')
s+='''
## Class-specific entity-history check

**Measured negative for a simple positive-risk rule.** Fit class-specific sender/receiver/device counts on full July–August; divide each class count by total entity count + 10, then take the maximum of the three entity rates for September. Evaluate isolated rows of that class versus legit, using the supplied full-train connectivity flags. AUCs: mule 0.4528, velocity 0.4560, layering 0.4612, SIM 0.4578, takeover 0.4795, phishing 0.5292, QR 0.4574, refund 0.4817, cashback 0.4867, loan 0.5059, collusion 0.4499.

These do not support “past same-class fraud means higher risk.” Below-chance AUC is an inverse association, not no information: it may reflect recurring merchant traffic, entity selection, or fraud rotation. This does not rule out a nonlinear, properly time-cross-fitted history feature. Prior “any fraud history hit rate” alone was insufficient to dismiss all target encoding. Evidence: `history.py`, `history.log`.
'''
open(p,'w').write(s)
