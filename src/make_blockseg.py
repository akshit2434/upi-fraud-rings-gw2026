"""hunt5 candidate: re-type every row in the October fraud device block by ordered segmentation (src/block_seg.py).
usage: .venv/bin/python src/make_blockseg.py --base submissions/v10_v6_octrules.csv --out submissions/v11_blockseg.csv
Emissions: model labels = v6 argmax (preds/v6_R1_bag_test_P.npy x weights); confusion C from Sep block rows
(v6 Sep val predictions /tmp/P_bag4_R1.npy, labels); rows/device + raw profile from all train-block rows."""
import argparse, numpy as np, pandas as pd, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from block_seg import *
ap = argparse.ArgumentParser(); ap.add_argument('--base', default='submissions/v10_v6_octrules.csv')
ap.add_argument('--out', default='submissions/v11_blockseg.csv'); ap.add_argument('--pen', type=float, default=1.0)
ap.add_argument('--w', type=float, default=1.0); ap.add_argument('--only', default='', help='comma list of zone classes to apply (default all)')
a = ap.parse_args()
D = 'data/upi-fraud-rings/'
cols = ['txn_id', 'timestamp', 'device_id', 'txn_type', 'receiver_type', 'amount']
tr = pd.read_csv(D + 'train.csv', usecols=cols + ['label']); te = pd.read_csv(D + 'test.csv', usecols=cols)
for x in (tr, te): x['dn'] = x.device_id.str[2:].astype(int)
TB0, TB1, OB0, OB1 = 423000, 428455, 473632, 474552   # train block (non-bot part), Oct block
trb = tr[(tr.dn >= TB0) & (tr.dn < TB1)].copy()
zt = trb[trb.label != 'legit'].groupby('dn').label.agg(lambda s: s.value_counts().index[0])
trb['zone'] = trb.dn.map(zt)
# Sep validation predictions of v6 (model trained Jul-Aug) -> confusion matrix C
P = np.load('/tmp/P_bag4_R1.npy'); ids = np.load('/tmp/P_bag4_R1_ids.npy', allow_pickle=True)
w6 = np.load('preds/v6_R1_bag_weights.npy')
sp = pd.DataFrame({'txn_id': ids, 'sp': np.array(LABS)[(P * w6).argmax(1)]})
s = trb.merge(sp, on='txn_id'); f = s[s.label != 'legit']
logC, _ = fit_emissions(f.label.values, f.sp.values, f.dn.values)
fl = trb[trb.label != 'legit']; _, logPn = fit_emissions(fl.label.values, fl.label.values, fl.dn.values)
z = trb[trb.zone.notna()]; logT = fit_profile(profile_key(z), z.zone.values)
# Oct block
Pt = np.load('preds/v6_R1_bag_test_P.npy'); tid = pd.read_csv('preds/v6_R1_bag_test_ids.csv').iloc[:, 0].values
v6 = pd.DataFrame({'txn_id': tid, 'v6': np.array(LABS)[(Pt * w6).argmax(1)]})
ob = te[(te.dn >= OB0) & (te.dn < OB1)].merge(v6, on='txn_id').sort_values('dn').reset_index(drop=True)
seg = segment(ob.dn.values, ob.v6.values, logC, logPn, switch_pen=a.pen, row_ll=profile_ll(profile_key(ob), logT), row_w=a.w)
ob['seg'] = ob.dn.map(seg)
base = pd.read_csv(a.base); m = base.set_index('id').label
new = ob.set_index('txn_id').seg
if a.only: new = new[new.isin(a.only.split(','))]
new = new[m.reindex(new.index) != 'merchant_collusion']   # seg has no reliable collusion zone (Sep 0/4) -> keep base
old = m.reindex(new.index)
print('zones:'); print(ob.groupby('seg').dn.agg(['min', 'max', 'size']).sort_values('min').to_string())
print('rows changed vs base:', int((old != new).sum())); print(pd.crosstab(old.rename('base'), new.rename('seg')).to_string())
m.loc[new.index] = new.values
out = m.reset_index(); out.columns = ['id', 'label']
assert len(out) == len(te) and out.id.is_unique and set(out.id) == set(te.txn_id)
out.to_csv(a.out, index=False); print('wrote', a.out, out.label.value_counts().to_dict())
