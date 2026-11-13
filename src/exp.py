"""Validation harness: train Jul-Aug, validate Sep (seed-s sampled frame). Reports argmax, same-set tuned and
cross-fitted macro-F1 (x-fit is the primary metric). Saves val probs with ids: /tmp/P_<tag>.npy + /tmp/P_<tag>_ids.npy."""
import sys, time, json, argparse
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, 'src')
from features import LABELS, load_cached, apply_drop
from common import macro_f1, tune_weights, xfit_f1, get_conn, clean_bad, apply_relabel

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--drop', default='')
ap.add_argument('--lr', type=float, default=0.05)
ap.add_argument('--leaves', type=int, default=63)
ap.add_argument('--mcs', type=int, default=40)
ap.add_argument('--ff', type=float, default=0.7)
ap.add_argument('--l2', type=float, default=1.0)
ap.add_argument('--legit_keep', type=float, default=0.25)
ap.add_argument('--iters', type=int, default=400)
ap.add_argument('--clean', default='iso7')
ap.add_argument('--relabel', default='')
ap.add_argument('--threads', type=int, default=10)
ap.add_argument('--tag', required=True)
a = ap.parse_args()
t0 = time.time()
X, meta = load_cached(a.seed)
X = apply_drop(X, a.drop)
ids = meta._txn_id.values
lab = meta._label.map({l: i for i, l in enumerate(LABELS)}).fillna(-1).astype(int).values
trn = ((meta._is_test == 0) & meta._month.isin(['2026-07', '2026-08'])).values
val = ((meta._is_test == 0) & (meta._month == '2026-09')).values
bad = clean_bad(lab, get_conn(ids, until='2026-09-01'), a.clean)       # Jul-Aug labels only
lab_tr = apply_relabel(lab, ids, a.relabel, mask=trn)                 # training labels only; val labels untouched
rng = np.random.default_rng(a.seed)
mask = trn & ~bad & ((lab_tr != 0) | (rng.random(len(X)) < a.legit_keep))
w = np.where(lab_tr[mask] == 0, 1 / a.legit_keep, 1.0)
params = dict(objective='multiclass', num_class=12, learning_rate=a.lr, num_leaves=a.leaves, min_child_samples=a.mcs,
              feature_fraction=a.ff, bagging_fraction=0.8, bagging_freq=1, lambda_l2=a.l2, verbose=-1, num_threads=a.threads, seed=a.seed)
m = lgb.train(params, lgb.Dataset(X[mask], lab_tr[mask], weight=w), a.iters)
P = m.predict(X[val]); y = lab[val]
np.save(f'/tmp/P_{a.tag}.npy', P); np.save(f'/tmp/P_{a.tag}_ids.npy', ids[val])
res = dict(argmax=round(macro_f1(y, P.argmax(1)), 4), sameset=round(tune_weights(P, y, iters=2)[1], 4), xfit=round(xfit_f1(P, y), 4))
print(a.tag, json.dumps(vars(a)), res, f'{time.time()-t0:.0f}s', flush=True)
imp = pd.Series(m.feature_importance('gain'), X.columns).sort_values(ascending=False); imp.to_csv(f'/tmp/imp_{a.tag}.csv')
