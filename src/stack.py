"""Two-stage model. Stage 1: LGB on base features, OOF via week-blocked folds. Stage 2: base features +
neighbour aggregates of stage-1 probabilities.
mode=val : base rows = Jul-Aug; evaluate stage 2 on Sep (Sep/test get stage-1 preds from model on all Jul-Aug).
mode=full: base rows = all train; predict test; writes preds/<name>_*.npy and submission.
"""
import sys, os, time, argparse, json
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, 'src')
from features import cache_path, load, build_features, sample_frame, neighbour_feats, LABELS
from common import macro_f1, tune_weights

ap = argparse.ArgumentParser()
ap.add_argument('--mode', default='val')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--k', type=int, default=4)
ap.add_argument('--it1', type=int, default=300)
ap.add_argument('--it2', default='100,200,400')
ap.add_argument('--lr', type=float, default=0.05)
ap.add_argument('--drop', default='')
ap.add_argument('--name', default='stack')
ap.add_argument('--val_P', default='')
args = ap.parse_args()
t00 = time.time()
LK = 0.25


def params(seed):
    return dict(objective='multiclass', num_class=len(LABELS), learning_rate=args.lr, num_leaves=63, min_child_samples=40,
                feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=10, seed=seed)


def fit(X, y, mask, it, seed):
    rng = np.random.default_rng(seed + 17)
    m = mask & ((y != 0) | (rng.random(len(y)) < LK))
    w = np.where(y[m] == 0, 1 / LK, 1.0)
    return lgb.train(params(seed), lgb.Dataset(X[m], y[m], weight=w), it)


tr, te, acc, mer = load()
df = sample_frame(tr, te, args.seed)
cache = cache_path(args.seed)
X = pd.read_parquet(cache).drop(columns=['_label', '_month', '_is_test', '_txn_id', 'cat_mismatch'], errors='ignore')
drop = [c for c in args.drop.split(',') if c]
X = X.drop(columns=[c for c in drop if c in X.columns])
assert len(X) == len(df)
y = df.label.map({l: i for i, l in enumerate(LABELS)}).fillna(-1).astype(int).values
month = df.timestamp.str[:7].values
is_test = df._is_test.values == 1
base = (~is_test) & (np.isin(month, ['2026-07', '2026-08']) if args.mode == 'val' else True)
week = ((pd.to_datetime(df.timestamp) - pd.Timestamp('2026-07-01')).dt.days.values // 7)
fold = week % args.k

# ---- stage 1 ----
P1 = np.zeros((len(df), len(LABELS)))
for f in range(args.k):
    m = fit(X, y, base & (fold != f), args.it1, args.seed + f)
    sel = base & (fold == f)
    P1[sel] = m.predict(X[sel])
    print('stage1 fold', f, round(time.time() - t00), 's', flush=True)
m = fit(X, y, base, args.it1, args.seed + 99)
rest = ~base
P1[rest] = m.predict(X[rest])
print('stage1 done', round(time.time() - t00), 's', flush=True)
if args.mode == 'val':
    sep = (~is_test) & (month == '2026-09')
    print('stage1 Sep F1 argmax', round(macro_f1(y[sep], P1[sep].argmax(1)), 4), 'tuned', round(tune_weights(P1[sep], y[sep], iters=2)[1], 4), flush=True)

# ---- stage 2 ----
NB = neighbour_feats(df, P1)
X2 = pd.concat([X, NB, pd.DataFrame(P1[:, 1:], columns=[f'p1_{k}' for k in range(1, len(LABELS))]).astype(np.float32)], axis=1)
its = [int(i) for i in args.it2.split(',')]
m2 = fit(X2, y, base, max(its), args.seed + 1000)
imp = pd.Series(m2.feature_importance('gain'), X2.columns).sort_values(ascending=False)
print((imp / imp.sum()).round(4).head(25).to_string())
if args.mode == 'val':
    for it in its:
        P2 = m2.predict(X2[sep], num_iteration=it)
        np.save(f'/tmp/P2_{args.name}_{it}.npy', P2)
        print('stage2 it', it, 'Sep F1 argmax', round(macro_f1(y[sep], P2.argmax(1)), 4), 'tuned', round(tune_weights(P2, y[sep], iters=2)[1], 4), flush=True)
    np.save('/tmp/y_val.npy', y[sep])
else:
    os.makedirs('preds', exist_ok=True); os.makedirs('submissions', exist_ok=True)
    P2 = m2.predict(X2[is_test], num_iteration=max(its))
    ids = df.txn_id.values[is_test]
    np.save(f'preds/{args.name}_s{args.seed}_test_P.npy', P2); pd.Series(ids).to_csv(f'preds/{args.name}_s{args.seed}_test_ids.csv', index=False)
    print('saved', args.name, args.seed)
print('total', round(time.time() - t00), 's')
