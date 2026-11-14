"""F4: does train under-reporting explain the Oct fraud excess? Unreported rate = legit-labelled share among rows that carry a
generator fraud fingerprint (independent of labels): train fraud device block, non-QR filled qr_vpa_mismatch marker,
and v6 very-confident Sep flags. Then predicted clean fraud share = observed share / (1 - unreported)."""
import numpy as np, pandas as pd
from load import R
tr = pd.read_csv(R + 'data/upi-fraud-rings/train.csv')
tr['m'] = tr.timestamp.str[:7]; tr = tr[tr.m.isin(['2026-07', '2026-08', '2026-09'])]
tr['dnum'] = tr.device_id.str[2:].astype(int)
fp = {'device block 420000-428454': tr.dnum.between(420000, 428454),
      'non-QR mismatch filled': (tr.txn_type != 'QR') & tr.qr_vpa_mismatch.notna(),
      'block & multi-row device': tr.dnum.between(420000, 428454) & (tr.groupby('device_id').txn_id.transform('size') > 1)}
for k, m in fp.items():
    x = tr[m]
    print(f'{k:30s} n={len(x):6d} legit share {100 * (x.label == "legit").mean():.1f}%  by month', x.groupby('m').label.apply(lambda s: round(100 * (s == 'legit').mean(), 1)).to_dict())
# v6 confident flags on Sep
from load import load
d = load(); s = d[d.grp == 'sep']
for th in [0.5, 0.8, 0.95, 0.99]:
    x = s[s.sp_pf >= th]; print(f'Sep v6 P(fraud)>={th}: n={len(x)} legit-labelled {100 * (x.y == "legit").mean():.1f}%')
obs = {m: 100 * (d[(d._month == m) & (d._is_test == 0)].fraud.mean()) for m in ['2026-07', '2026-08', '2026-09']}
print('observed fraud share 51% frame', {k: round(v, 3) for k, v in obs.items()})
for u in [0.08, 0.10, 0.12]:
    print(f'unreported {u:.0%}: clean share = {np.mean(list(obs.values())) / (1 - u):.2f}%  -> test rows {np.mean(list(obs.values())) / (1 - u) / 100 * 260582:.0f} (probe 4.07% = {0.0407 * 260582:.0f})')
