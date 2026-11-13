"""Label-structure flag: a fraud row is 'connected' if it shares sender/receiver/device with another row of the SAME
label within +-6h, or is a same-label chain hop within 6h. Labels only (no test).
usage: connectivity.py [--until YYYY-MM-DD]  -> data/conn.parquet (all train) or data/conn_until_<date>.parquet
(--until restricts to labels strictly before that date, so validation cleaning never sees validation labels)."""
import argparse, pandas as pd, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--until', default=''); a = ap.parse_args()
H = 6 * 3600
tr = pd.read_csv('data/upi-fraud-rings/train.csv', usecols=['txn_id', 'timestamp', 'sender_id', 'receiver_id', 'device_id', 'label'])
if a.until:
    tr = tr[tr.timestamp < a.until]
tr['t'] = (pd.to_datetime(tr.timestamp) - pd.Timestamp('2026-07-01')).dt.total_seconds().astype(int)
conn = pd.Series(False, index=tr.txn_id)
for lab in tr.label.unique():
    if lab == 'legit':
        continue
    f = tr[tr.label == lab][['txn_id', 't', 'sender_id', 'receiver_id', 'device_id']]
    for k in ['sender_id', 'receiver_id', 'device_id']:
        m = f.merge(f, on=k, suffixes=('', '_o')); m = m[(m.txn_id != m.txn_id_o) & ((m.t - m.t_o).abs() < H)]
        conn[m.txn_id.unique()] = True
    for x, y, sg in [('receiver_id', 'sender_id', 1), ('sender_id', 'receiver_id', -1)]:
        m = f.merge(f, left_on=x, right_on=y, suffixes=('', '_o')); m = m[(sg * (m.t_o - m.t)).between(0, H)]
        conn[m.txn_id.unique()] = True
out = conn.rename('conn').reset_index()
path = f'data/conn_until_{a.until}.parquet' if a.until else 'data/conn.parquet'
out.to_parquet(path)
print(path, out.conn.sum(), 'connected fraud rows of', len(out))
