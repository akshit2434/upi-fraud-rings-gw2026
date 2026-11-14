"""Entity label-history features (target encoding across months).
For a row in month M, counts fraud labels (total and per class) of the same receiver / sender / device in the ALLOWED
label months other than M. Variants:
  histv : allowed = Jul, Aug            (validation: Sep labels never used; Sep rows see Jul+Aug)
  histf : allowed = Jul, Aug, Sep       (full training + test; test rows (Oct) see all three months)
Writes /tmp/<variant>_<FEATURE_VERSION>_s<seed>.parquet aligned to features.load_cached(seed) row order.
usage: hist_feats.py --seeds 0,1,2,3"""
import sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, 'src')
from features import load, sample_frame, LABELS, FEATURE_VERSION

MONTHS = ['2026-07', '2026-08', '2026-09']
ENTS = {'r': 'receiver_id', 's': 'sender_id', 'd': 'device_id'}


def build(tr, te, allowed):
    al = pd.concat([tr[['txn_id', 'timestamp', 'sender_id', 'receiver_id', 'device_id', 'receiver_type']],
                    te[['txn_id', 'timestamp', 'sender_id', 'receiver_id', 'device_id', 'receiver_type']]], ignore_index=True)
    al['m'] = al.timestamp.str[:7]
    L = tr[tr.timestamp.str[:7].isin(allowed)][['timestamp', 'sender_id', 'receiver_id', 'device_id', 'label']].copy()
    L['m'] = L.timestamp.str[:7]
    out = pd.DataFrame({'_txn_id': al.txn_id.values})
    for k, col in ENTS.items():
        # per (entity, month): row count, fraud count, per-class counts
        g = pd.crosstab([L[col], L.m], L.label)
        for c in LABELS:
            if c not in g.columns:
                g[c] = 0
        g = g[LABELS]
        tot = g.groupby(level=0).sum()  # entity totals over allowed months
        key_m = pd.MultiIndex.from_arrays([al[col].values, al.m.values])
        own = g.reindex(key_m).fillna(0).values          # same-month contribution (to subtract)
        allm = tot.reindex(al[col].values).fillna(0).values
        other = allm - own                                 # allowed months other than the row's month
        n = other.sum(1); f = n - other[:, 0]
        out[f'h{k}_n'] = n
        out[f'h{k}_fraud'] = f
        out[f'h{k}_frate'] = np.where(n > 0, f / np.maximum(n, 1), np.nan)
        for i, c in enumerate(LABELS[1:], start=1):
            out[f'h{k}_{c[:6]}'] = other[:, i]
    p2p = (al.receiver_type.values == 'P2P')
    for c in [x for x in out.columns if x.startswith('hr_')]:
        out[c] = np.where(p2p, out[c], np.nan)  # merchant receivers: history is meaningless noise
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', default='0,1,2,3'); a = ap.parse_args()
    tr, te, _, _ = load()
    for var, allowed in [('histv', MONTHS[:2]), ('histf', MONTHS)]:
        H = build(tr, te, allowed).set_index('_txn_id')
        for s in [x for x in a.seeds.split(',') if x]:
            s = int(s) if s != 'full' else s
            ids = (sample_frame(tr, te, 0, rate=1.0) if s == 'full' else sample_frame(tr, te, s)).txn_id.values
            F = H.reindex(ids).astype(np.float32); F.index.name = '_txn_id'; F = F.reset_index()
            F.to_parquet(f'/tmp/{var}_{FEATURE_VERSION}_s{s}.parquet')
        print(var, H.shape, 'nonzero r-fraud share', round((H.hr_fraud > 0).mean(), 4), flush=True)
