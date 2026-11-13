"""Write a submission from saved test probs, with class weights tuned on validation probs (aligned by txn id).
usage: make_sub.py --pred preds/<name>_test_P.npy --val_P /tmp/P_<tag>.npy [--wmode full|none] [--scale s] --out <name>"""
import sys, argparse, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from features import LABELS
from common import tune_weights, macro_f1, xfit_f1
ap = argparse.ArgumentParser()
ap.add_argument('--pred', required=True); ap.add_argument('--val_P', default=''); ap.add_argument('--wmode', default='full')
ap.add_argument('--scale', type=float, default=1.0); ap.add_argument('--out', required=True)
a = ap.parse_args()
P = np.load(a.pred); ids = pd.read_csv(a.pred.replace('_test_P.npy', '_test_ids.csv')).iloc[:, 0].values
wts = np.ones(len(LABELS))
if a.wmode != 'none':
    Pv = np.load(a.val_P); vids = np.load(a.val_P.replace('.npy', '_ids.npy'), allow_pickle=True)
    lab = pd.read_csv('data/upi-fraud-rings/train.csv', usecols=['txn_id', 'label']).set_index('txn_id').label
    yv = lab.reindex(vids).map({l: i for i, l in enumerate(LABELS)}).values.astype(int)
    wts, f = tune_weights(Pv, yv, iters=3)
    print('val same-set F1', round(f, 4), '| x-fit', round(xfit_f1(Pv, yv), 4))
wts[1:] *= a.scale
print('weights', np.round(wts, 2).tolist())
te = pd.read_csv('data/upi-fraud-rings/test.csv', usecols=['txn_id'])
pred = pd.Series(np.array(LABELS)[(P * wts).argmax(1)], index=ids)
sub = pd.DataFrame({'id': te.txn_id, 'label': pred.reindex(te.txn_id).values})
assert sub.label.notna().all() and len(sub) == len(te) and sub.id.is_unique
sub.to_csv(f'submissions/{a.out}.csv', index=False); np.save(f'preds/{a.out}_weights.npy', wts)
print(a.out, 'fraud share', round((sub.label != 'legit').mean(), 4))
