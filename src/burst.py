"""For every train row, count same-class fraud neighbours (per class) sharing sender/receiver/device within +-W,
plus chain links. Used to find unreported fraud (legit rows embedded in fraud bursts). Full-density train, labels only."""
import pandas as pd, numpy as np, sys
W = int(sys.argv[1]) if len(sys.argv) > 1 else 3600
tr = pd.read_parquet('/tmp/tr.parquet')[['txn_id', 't', 'sender_id', 'receiver_id', 'device_id', 'label']]
fr = tr[tr.label != 'legit']
rows = []
for k in ['sender_id', 'receiver_id', 'device_id']:
    if k == 'receiver_id':
        fk = fr[fr.receiver_id.str.startswith('AC')]
    else:
        fk = fr
    m = tr.merge(fk[[k, 't', 'label', 'txn_id']], on=k, suffixes=('', '_f'))
    m = m[(m.txn_id != m.txn_id_f) & ((m.t - m.t_f).abs() <= W)]
    rows.append(m[['txn_id', 'label_f', 'txn_id_f']])
# chain: row's receiver is fraud row's sender shortly after / row's sender was paid by fraud row shortly before
m = tr.merge(fr[['sender_id', 't', 'label', 'txn_id']], left_on='receiver_id', right_on='sender_id', suffixes=('', '_f'))
m = m[(m.t_f - m.t).between(0, W)]; rows.append(m[['txn_id', 'label_f', 'txn_id_f']])
m = tr.merge(fr[['receiver_id', 't', 'label', 'txn_id']], left_on='sender_id', right_on='receiver_id', suffixes=('', '_f'))
m = m[(m.t - m.t_f).between(0, W)]; rows.append(m[['txn_id', 'label_f', 'txn_id_f']])
nb = pd.concat(rows).drop_duplicates(['txn_id', 'txn_id_f'])
cnt = nb.groupby(['txn_id', 'label_f']).size().unstack(fill_value=0)
cnt.to_parquet(f'data/burst_nb_{W}.parquet')
lab = tr.set_index('txn_id').label
top = cnt.idxmax(axis=1); topn = cnt.max(axis=1)
d = pd.DataFrame({'label': lab.reindex(cnt.index), 'top': top, 'n': topn})
print('rows with any fraud neighbour:', len(d))
for thr in [1, 2, 3, 5]:
    x = d[d.n >= thr]
    print(f'n>={thr}: legit rows {int((x.label=="legit").sum())}, fraud rows {int((x.label!="legit").sum())}, of which label==top {(x.label==x.top).sum()}')
x = d[(d.n >= 2) & (d.label == 'legit')]
print('legit rows with >=2 same-class nbrs, by class:', x.top.value_counts().to_dict())
x = d[(d.n >= 2) & (d.label != 'legit')]
print('fraud rows with >=2 nbrs: label agrees with top', round((x.label == x.top).mean(), 3))
