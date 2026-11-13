"""Train on ALL training months for each sampling seed, predict test, average over seeds.
Writes preds/<name>_test_P.npy + preds/<name>_test_ids.csv (+ preds/<name>_manifest.json). Use make_sub.py for the CSV."""
import sys, os, time, json, argparse
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, 'src')
from features import LABELS, FEATURE_VERSION, load_cached, apply_drop
from common import get_conn, clean_bad, apply_relabel

ap = argparse.ArgumentParser()
ap.add_argument('--name', required=True)
ap.add_argument('--seeds', default='0')
ap.add_argument('--iters', type=int, default=400)
ap.add_argument('--lr', type=float, default=0.05)
ap.add_argument('--leaves', type=int, default=63)
ap.add_argument('--mcs', type=int, default=40)
ap.add_argument('--ff', type=float, default=0.7)
ap.add_argument('--l2', type=float, default=1.0)
ap.add_argument('--legit_keep', type=float, default=0.25)
ap.add_argument('--drop', default='')
ap.add_argument('--clean', default='iso7')
ap.add_argument('--relabel', default='')
ap.add_argument('--threads', type=int, default=10)
a = ap.parse_args()
os.makedirs('preds', exist_ok=True)
Ps, ids0 = [], None
for seed in [int(s) for s in a.seeds.split(',')]:
    t0 = time.time()
    X, meta = load_cached(seed)
    X = apply_drop(X, a.drop)
    ids = meta._txn_id.values
    lab = meta._label.map({l: i for i, l in enumerate(LABELS)}).fillna(-1).astype(int).values
    trn = (meta._is_test == 0).values
    bad = clean_bad(lab, get_conn(ids), a.clean)
    lab = apply_relabel(lab, ids, a.relabel, mask=trn)
    rng = np.random.default_rng(seed)
    mask = trn & ~bad & ((lab != 0) | (rng.random(len(X)) < a.legit_keep))
    w = np.where(lab[mask] == 0, 1 / a.legit_keep, 1.0)
    params = dict(objective='multiclass', num_class=12, learning_rate=a.lr, num_leaves=a.leaves, min_child_samples=a.mcs,
                  feature_fraction=a.ff, bagging_fraction=0.8, bagging_freq=1, lambda_l2=a.l2, verbose=-1, num_threads=a.threads, seed=seed)
    m = lgb.train(params, lgb.Dataset(X[mask], lab[mask], weight=w), a.iters)
    tm = (meta._is_test == 1).values
    tid = ids[tm]; o = np.argsort(tid)
    if ids0 is None: ids0 = tid[o]
    assert (tid[o] == ids0).all()
    Ps.append(m.predict(X[tm])[o])
    print('seed', seed, 'dropped', int((bad & trn).sum()), 'done', round(time.time() - t0), 's', flush=True)
P = np.mean(Ps, axis=0)
np.save(f'preds/{a.name}_test_P.npy', P); pd.Series(ids0).to_csv(f'preds/{a.name}_test_ids.csv', index=False)
json.dump(dict(vars(a), feature_version=FEATURE_VERSION, classes=LABELS, columns=list(X.columns)), open(f'preds/{a.name}_manifest.json', 'w'), indent=1)
print('saved', a.name)
