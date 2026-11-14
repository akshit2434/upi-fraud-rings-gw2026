"""F2: extreme-tail excess scan. For each feature x cell (txn_type x receiver_type, plus ALL), thresholds at Sep-unflagged
quantiles (upper .99/.998, lower .002/.01 when distinct): count Oct-unflagged vs Sep-unflagged rows beyond. Rank by excess & z."""
import numpy as np, pandas as pd
from load import load
d = load()
d['cell'] = d.tt + '>' + d.rt
F = [c for c in d.columns[:155] if not c.startswith('_') and c not in ('cat_mismatch',)]
s0 = d[(d.grp == 'sep') & (~d.flag)]; o0 = d[(d.grp == 'oct') & (~d.flag)]
scale = len(o0) / len(s0)
rows = []
for cell in ['ALL'] + list(s0.cell.value_counts().index[:8]):
    s = s0 if cell == 'ALL' else s0[s0.cell == cell]; o = o0 if cell == 'ALL' else o0[o0.cell == cell]
    for f in F:
        sv = s[f].values.astype(float); ov = o[f].values.astype(float)
        if np.nanstd(sv) == 0: continue
        for q in [0.99, 0.998, 0.002, 0.01]:
            th = np.nanquantile(sv, q)
            if q > 0.5: ns = np.sum(sv > th); no = np.sum(ov > th)
            else: ns = np.sum(sv < th); no = np.sum(ov < th)
            if ns + no < 20: continue
            exp = ns * scale
            sub = s[(s[f] > th) if q > 0.5 else (s[f] < th)]
            rows.append({'cell': cell, 'f': f, 'q': q, 'th': round(th, 3), 'sep': ns, 'oct': no, 'exc': round(no - exp), 'z': round((no - exp) / np.sqrt(exp + no + 1) * 1.414, 1),
                         'sepfraud%': round(100 * sub.fraud.mean(), 1)})
r = pd.DataFrame(rows).sort_values('z', ascending=False)
r.to_csv('/tmp/hunt7/f2_tails.csv', index=False)
pd.set_option('display.width', 200); pd.set_option('display.max_rows', 200)
print(r.head(70).to_string())
