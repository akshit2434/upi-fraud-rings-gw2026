"""For Sep val rows (seed-0 frame): build same-class labelled links within the SAMPLED frame (sender/device/P2P-receiver/chain, 6h),
then for connected fraud rows the model gets wrong, check what the model predicts for their linked partners."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from features import LABELS, load, sample_frame
from common import tune_weights
tr, te, acc, mer = load(); df = sample_frame(tr, te, 0)
df['t'] = (pd.to_datetime(df.timestamp) - pd.Timestamp('2026-07-01')).dt.total_seconds().astype(int)
sep = ((df._is_test == 0) & (df.timestamp.str[:7] == '2026-09')).values
v = df[sep].reset_index(drop=True)
y = v.label.map({l: i for i, l in enumerate(LABELS)}).values
P = np.load(sys.argv[1] if len(sys.argv) > 1 else '/tmp/P_bag4_noip.npy'); w, _ = tune_weights(P, y, iters=2)
pr = (P * w).argmax(1); v['y'] = y; v['pr'] = pr; v['i'] = np.arange(len(v))
H = 6 * 3600
links = []
f = v[v.y > 0]
for a, b in [('sender_id', 'sender_id'), ('device_id', 'device_id'), ('receiver_id', 'receiver_id'), ('receiver_id', 'sender_id'), ('sender_id', 'receiver_id')]:
    L = f[['i', 't', a, 'y']]; R = v[['i', 't', b, 'y', 'pr']].rename(columns={'i': 'j', 't': 'tj', b: a, 'y': 'yj', 'pr': 'prj'})
    if a == 'receiver_id' and b == 'receiver_id': L = L[L.receiver_id.str.startswith('AC')]
    m = L.merge(R, on=a); m = m[(m.i != m.j) & ((m.t - m.tj).abs() <= H)]
    links.append(m[['i', 'j', 'y', 'yj', 'prj']])
lk = pd.concat(links).drop_duplicates(['i', 'j'])
same = lk[lk.y == lk.yj]
conn_rows = same.i.unique()
print('connected fraud rows (sampled frame):', len(conn_rows))
c = v.loc[conn_rows]
miss = c[c.pr == 0]; wrong = c[(c.pr > 0) & (c.pr != c.y)]
print('called legit', len(miss), '| wrong type', len(wrong))
g = same.groupby('i')
partner_right = g.apply(lambda d: (d.prj == d.y).any())
partner_anyfraud = g.apply(lambda d: (d.prj > 0).any())
print('missed rows: a same-class partner predicted correctly:', round(partner_right.reindex(miss.i).mean(), 3), '| partner predicted any fraud:', round(partner_anyfraud.reindex(miss.i).mean(), 3))
print('wrong-type rows: partner predicted correct type:', round(partner_right.reindex(wrong.i).mean(), 3))
print('missed by class:', miss.y.map(dict(enumerate(LABELS))).value_counts().to_dict())
print('wrong type pairs (true->pred):', (wrong.y.map(dict(enumerate(LABELS))) + '->' + wrong.pr.map(dict(enumerate(LABELS)))).value_counts().head(10).to_dict())
# legit neighbours of correctly-predicted fraud: how many legit-labelled rows linked would be FPs if propagated
lg = lk[(lk.yj == 0)]
print('legit-labelled rows linked to fraud rows:', lg.j.nunique(), ' (would be FPs under naive propagation)')
v[['txn_id', 'y', 'pr']].to_parquet('/tmp/event_v.parquet'); lk.to_parquet('/tmp/event_links.parquet')
