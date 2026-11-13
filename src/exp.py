"""Fast experiment harness: cached features, train Jul-Aug, validate Sep, F1 at checkpoints."""
import sys, os, time, json, argparse
import numpy as np, pandas as pd, lightgbm as lgb

sys.path.insert(0, 'src')
from features import load, build_features, sample_frame, LABELS
from common import macro_f1, tune_weights

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--drop', default='')
ap.add_argument('--keep_only', default='')
ap.add_argument('--lr', type=float, default=0.05)
ap.add_argument('--leaves', type=int, default=63)
ap.add_argument('--mcs', type=int, default=40)
ap.add_argument('--ff', type=float, default=0.7)
ap.add_argument('--l2', type=float, default=1.0)
ap.add_argument('--legit_keep', type=float, default=0.25)
ap.add_argument('--iters', default='50,100,200,400')
ap.add_argument('--rebuild', action='store_true')
ap.add_argument('--cw', default='none')  # none | sqrt | bal
ap.add_argument('--tag', default='')
ap.add_argument('--clean', default='none')  # none | iso7 | isoall | iso7relabel
ap.add_argument('--relabel', default='')  # data/relabel_<name>.parquet applied to TRAINING labels; eval reported on raw and relabelled Sep
args = ap.parse_args()

cache = f'/tmp/feat_s{args.seed}.parquet'
if args.rebuild or not os.path.exists(cache):
    tr, te, acc, mer = load()
    df = sample_frame(tr, te, args.seed)
    X = build_features(df, acc, mer)
    X['_label'] = df.label.values
    X['_month'] = df.timestamp.str[:7].values
    X['_is_test'] = df._is_test.values
    X['_txn_id'] = df.txn_id.values
    X.to_parquet(cache)
X = pd.read_parquet(cache)
meta = X[['_label', '_month', '_is_test', '_txn_id']]
X = X.drop(columns=['_label', '_month', '_is_test', '_txn_id', 'cat_mismatch'], errors='ignore')
drop = [c for c in args.drop.split(',') if c]
X = X.drop(columns=[c for c in X.columns if any(c == d or (d.endswith('*') and c.startswith(d[:-1])) for d in drop)])
lab = meta._label.map({l: i for i, l in enumerate(LABELS)})
trn = (meta._is_test == 0) & meta._month.isin(['2026-07', '2026-08'])
val = (meta._is_test == 0) & (meta._month == '2026-09')
rng = np.random.default_rng(args.seed)
conn = pd.read_parquet('data/conn.parquet').set_index('txn_id').conn.reindex(meta._txn_id.values).fillna(False).values.astype(bool)
STRUCT = [LABELS.index(c) for c in ['mule_account', 'velocity_bot', 'layering_chain', 'sim_swap', 'account_takeover', 'merchant_collusion', 'fake_cashback']]
iso = (lab.fillna(0).values > 0) & ~conn
if args.clean in ('iso7', 'iso7relabel'):
    bad = iso & np.isin(lab.fillna(0).values, STRUCT)
elif args.clean == 'isoall':
    bad = iso
else:
    bad = np.zeros(len(X), bool)
lab_eval = lab.copy()
if args.clean == 'iso7relabel':
    lab = lab.where(~bad, 0); bad = np.zeros(len(X), bool)
rl_eval = lab_eval.copy()
if args.relabel:
    rl = pd.read_parquet(args.relabel).set_index('txn_id').new_label
    newv = rl.reindex(meta._txn_id.values).values
    hasr = ~np.isnan(newv)
    lab = lab.where(~hasr, newv); rl_eval = rl_eval.where(~hasr, newv)
    print('relabelled', int((hasr & trn.values).sum()), 'train rows', int((hasr & val.values).sum()), 'val rows (only in denoised eval)', flush=True)
print('cleaning', args.clean, 'removed', int((bad & trn).sum()), 'train rows', flush=True)
mask = trn & ~bad & ((lab != 0) | (rng.random(len(X)) < args.legit_keep))
yl = lab[mask].values
w = np.where(yl == 0, 1 / args.legit_keep, 1.0)
if args.cw != 'none':
    cnt = np.bincount(yl, weights=w, minlength=len(LABELS))
    cwt = (cnt.max() / cnt) ** (0.5 if args.cw == 'sqrt' else 1.0)
    cwt = cwt / cwt[0]
    w = w * cwt[yl] / np.where(yl == 0, 1, 1)
params = dict(objective='multiclass', num_class=len(LABELS), learning_rate=args.lr, num_leaves=args.leaves,
              min_child_samples=args.mcs, feature_fraction=args.ff, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=args.l2, verbose=-1, num_threads=10, seed=args.seed)
its = [int(i) for i in args.iters.split(',')]
t0 = time.time()
m = lgb.train(params, lgb.Dataset(X[mask], yl, weight=w), max(its))
y = lab_eval[val].values.astype(int)
res = {}
for it in its:
    P = m.predict(X[val], num_iteration=it)
    wts, best = tune_weights(P, y, iters=2)
    keep = (y == 0) | conn[val.values]
    wp, bp = tune_weights(P[keep], y[keep], iters=2)
    yd = rl_eval[val].values.astype(int)
    wd, bd = tune_weights(P, yd, iters=2)
    res[it] = dict(denoised=(round(macro_f1(yd, P.argmax(1)), 4), round(bd, 4)), full=(round(macro_f1(y, P.argmax(1)), 4), round(best, 4)), proxy=(round(macro_f1(y[keep], P[keep].argmax(1)), 4), round(bp, 4)),
                   full_w_proxy=round(macro_f1(y, (P * wp).argmax(1)), 4))
    np.save(f'/tmp/P_{args.tag or "x"}_{it}.npy', P)
np.save('/tmp/y_val.npy', y)
print(args.tag, json.dumps(vars(args)), '| F1 (argmax, tuned) by iter:', res, f'| {time.time()-t0:.0f}s', flush=True)
if args.tag:
    imp = pd.Series(m.feature_importance('gain'), X.columns).sort_values(ascending=False)
    imp.to_csv(f'/tmp/imp_{args.tag}.csv')
