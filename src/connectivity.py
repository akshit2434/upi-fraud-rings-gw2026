"""Label-structure flag on the FULL train (labels only, no test): a fraud row is 'connected' if it shares
sender/receiver/device with another row of the SAME label within +-6h, or is a same-label chain hop
(its receiver pays / its sender was paid, same label, within 6h). Output: data/conn.parquet (txn_id, conn)."""
import pandas as pd, numpy as np
H = 6 * 3600
tr = pd.read_csv('data/upi-fraud-rings/train.csv', usecols=['txn_id', 'timestamp', 'sender_id', 'receiver_id', 'device_id', 'label'])
tr['t'] = (pd.to_datetime(tr.timestamp) - pd.Timestamp('2026-07-01')).dt.total_seconds().astype(int)
conn = pd.Series(False, index=tr.txn_id)
for lab in tr.label.unique():
    if lab == 'legit':
        continue
    f = tr[tr.label == lab][['txn_id', 't', 'sender_id', 'receiver_id', 'device_id']]
    for k in ['sender_id', 'receiver_id', 'device_id']:
        m = f.merge(f, on=k, suffixes=('', '_o')); m = m[(m.txn_id != m.txn_id_o) & ((m.t - m.t_o).abs() < H)]
        conn[m.txn_id.unique()] = True
    for a, b, sg in [('receiver_id', 'sender_id', 1), ('sender_id', 'receiver_id', -1)]:
        m = f.merge(f, left_on=a, right_on=b, suffixes=('', '_o')); m = m[(sg * (m.t_o - m.t)).between(0, H)]
        conn[m.txn_id.unique()] = True
out = conn.rename('conn').reset_index()
out.to_parquet('data/conn.parquet')
print(tr.assign(conn=conn.values).groupby('label').conn.mean().round(3).to_string())
