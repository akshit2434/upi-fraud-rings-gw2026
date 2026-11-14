"""A/B: v6 predictions, plus any row with P(fraud) = 1 - P(legit) > T relabelled to its best weighted fraud class.
usage: make_flagmore.py --pred preds/v6_R1_bag_test_P.npy --weights preds/v6_R1_bag_weights.npy --t 0.02 --out NAME"""
import sys, argparse, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from features import LABELS
ap = argparse.ArgumentParser(); ap.add_argument('--pred'); ap.add_argument('--weights'); ap.add_argument('--t', type=float); ap.add_argument('--out')
a = ap.parse_args()
P = np.load(a.pred); w = np.load(a.weights); ids = pd.read_csv(a.pred.replace('_test_P.npy', '_test_ids.csv')).iloc[:, 0].values
pr = (P * w).argmax(1); fc = (P[:, 1:] * w[1:]).argmax(1) + 1
pr = np.where((1 - P[:, 0] > a.t) & (pr == 0), fc, pr)
te = pd.read_csv('data/upi-fraud-rings/test.csv', usecols=['txn_id'])
sub = pd.DataFrame({'id': te.txn_id, 'label': pd.Series(np.array(LABELS)[pr], index=ids).reindex(te.txn_id).values})
assert sub.label.notna().all(); sub.to_csv(f'submissions/{a.out}.csv', index=False)
print(a.out, 'fraud share', round((sub.label != 'legit').mean(), 4), sub.label.value_counts().to_dict())
