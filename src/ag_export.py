"""Export the exact exp2 training/val/test matrices for AutoGluon (separate env). Writes /tmp/ag/<mode>_{train,pred}.parquet.
mode=val: train Jul-Aug (seed-0 frame), pred = seed-0 Sep rows. mode=full: train all months, pred = test rows."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from features import LABELS, load_cached, apply_drop
from common import get_conn, clean_bad, apply_relabel
mode = sys.argv[1]; os.makedirs('/tmp/ag', exist_ok=True)
X, meta = load_cached(0); X = apply_drop(X, 'ip_rank_month'); ids = meta._txn_id.values
lab0 = meta._label.map({l: i for i, l in enumerate(LABELS)}).fillna(-1).astype(int).values
if mode == 'val':
    trn = ((meta._is_test == 0) & meta._month.isin(['2026-07', '2026-08'])).values; conn = get_conn(ids, until='2026-09-01')
    pm = ((meta._is_test == 0) & (meta._month == '2026-09')).values
else:
    trn = (meta._is_test == 0).values; conn = get_conn(ids); pm = (meta._is_test == 1).values
bad = clean_bad(lab0, conn, 'iso7'); lab = apply_relabel(lab0, ids, 'data/relabel_R1.parquet', mask=trn)
rng = np.random.default_rng(0); mask = trn & ~bad & ((lab != 0) | (rng.random(len(X)) < 0.25))
T = X[mask].copy(); T['label'] = lab[mask]; T['w'] = np.where(lab[mask] == 0, 4.0, 1.0)
T.to_parquet(f'/tmp/ag/{mode}_train.parquet')
Pr = X[pm].copy(); Pr['_txn_id'] = ids[pm]; Pr.to_parquet(f'/tmp/ag/{mode}_pred.parquet')
print(mode, T.shape, Pr.shape)
