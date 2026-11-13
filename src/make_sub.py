"""Write a submission from saved test probabilities with class weights tuned on a val prob file.
usage: make_sub.py --pred preds/<name>_test_P.npy --val_P /tmp/P_x.npy --wmode full|proxy|none [--scale s] --out name"""
import sys, argparse, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from features import LABELS
from common import tune_weights, macro_f1
ap = argparse.ArgumentParser()
ap.add_argument('--pred', required=True); ap.add_argument('--val_P', default=''); ap.add_argument('--wmode', default='full')
ap.add_argument('--scale', type=float, default=1.0)  # multiply all fraud weights (LB calibration)
ap.add_argument('--out', required=True)
a = ap.parse_args()
P = np.load(a.pred); ids = pd.read_csv(a.pred.replace('_test_P.npy', '_test_ids.csv')).iloc[:, 0].values
wts = np.ones(len(LABELS))
if a.wmode != 'none':
    meta = pd.read_parquet('/tmp/feat_s0.parquet', columns=['_label', '_month', '_is_test', '_txn_id'])
    v = (meta._is_test == 0) & (meta._month == '2026-09')
    yv = meta._label[v].map({l: i for i, l in enumerate(LABELS)}).values.astype(int)
    conn = pd.read_parquet('data/conn.parquet').set_index('txn_id').conn.reindex(meta._txn_id[v].values).fillna(False).values.astype(bool)
    Pv = np.load(a.val_P); assert len(Pv) == len(yv)
    keep = ((yv == 0) | conn) if a.wmode == 'proxy' else np.ones(len(yv), bool)
    wts, f = tune_weights(Pv[keep], yv[keep], iters=3)
    print(a.wmode, 'val tuned F1', round(f, 4), '| full-val F1 with these weights', round(macro_f1(yv, (Pv * wts).argmax(1)), 4))
wts[1:] *= a.scale
print('weights', np.round(wts, 2).tolist())
te = pd.read_csv('data/upi-fraud-rings/test.csv', usecols=['txn_id'])
pred = pd.Series(np.array(LABELS)[(P * wts).argmax(1)], index=ids)
sub = pd.DataFrame({'id': te.txn_id, 'label': pred.reindex(te.txn_id).values})
assert sub.label.notna().all() and len(sub) == len(te) and sub.id.is_unique
sub.to_csv(f'submissions/{a.out}.csv', index=False)
np.save(f'preds/{a.out}_weights.npy', wts)
print(a.out, 'fraud share', round((sub.label != 'legit').mean(), 4))
