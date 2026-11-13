"""Train on all training months and predict test. Averages over sampling seeds.

Writes probabilities to preds/<name>_test_P.npy and a submission using per-class weights
(tuned on the Sep validation run from exp.py, passed via --weights_from <P_file>).
"""
import sys, os, time, argparse
import numpy as np, pandas as pd, lightgbm as lgb

sys.path.insert(0, 'src')
from features import load, build_features, sample_frame, LABELS
from common import tune_weights, macro_f1

ap = argparse.ArgumentParser()
ap.add_argument('--name', required=True)
ap.add_argument('--seeds', default='0')
ap.add_argument('--iters', type=int, default=400)
ap.add_argument('--lr', type=float, default=0.05)
ap.add_argument('--leaves', type=int, default=63)
ap.add_argument('--mcs', type=int, default=40)
ap.add_argument('--legit_keep', type=float, default=0.25)
ap.add_argument('--drop', default='')
ap.add_argument('--val_P', default='')  # validation probs to tune class weights on
ap.add_argument('--clean', default='none')
ap.add_argument('--wmode', default='full,proxy')
ap.add_argument('--relabel', default='')
args = ap.parse_args()
os.makedirs('preds', exist_ok=True); os.makedirs('submissions', exist_ok=True)

tr, te, acc, mer = load()
Ps = []
for seed in [int(s) for s in args.seeds.split(',')]:
    t0 = time.time()
    cache = f'/tmp/feat_s{seed}.parquet'
    if not os.path.exists(cache):
        df = sample_frame(tr, te, seed)
        X = build_features(df, acc, mer)
        X['_label'] = df.label.values; X['_month'] = df.timestamp.str[:7].values
        X['_is_test'] = df._is_test.values; X['_txn_id'] = df.txn_id.values
        X.to_parquet(cache)
    X = pd.read_parquet(cache)
    meta = X[['_label', '_month', '_is_test', '_txn_id']]
    X = X.drop(columns=['_label', '_month', '_is_test', '_txn_id', 'cat_mismatch'], errors='ignore')
    drop = [c for c in args.drop.split(',') if c]
    X = X.drop(columns=[c for c in drop if c in X.columns])
    lab = meta._label.map({l: i for i, l in enumerate(LABELS)})
    trn = meta._is_test == 0
    rng = np.random.default_rng(seed)
    conn = pd.read_parquet('data/conn.parquet').set_index('txn_id').conn.reindex(meta._txn_id.values).fillna(False).values.astype(bool)
    STRUCT = [LABELS.index(c) for c in ['mule_account', 'velocity_bot', 'layering_chain', 'sim_swap', 'account_takeover', 'merchant_collusion', 'fake_cashback']]
    bad = np.zeros(len(X), bool)
    if args.clean == 'iso7':
        bad = (lab.fillna(0).values > 0) & ~conn & np.isin(lab.fillna(0).values, STRUCT)
    if args.relabel:
        rl = pd.read_parquet(args.relabel).set_index('txn_id').new_label.reindex(meta._txn_id.values).values
        hasr = ~np.isnan(rl); lab = lab.where(~hasr, rl)
        print('relabelled', int(hasr.sum()), flush=True)
    print('clean', args.clean, 'dropped', int((bad & trn.values).sum()), flush=True)
    mask = trn & ~bad & ((lab != 0) | (rng.random(len(X)) < args.legit_keep))
    yl = lab[mask].values
    w = np.where(yl == 0, 1 / args.legit_keep, 1.0)
    params = dict(objective='multiclass', num_class=len(LABELS), learning_rate=args.lr, num_leaves=args.leaves,
                  min_child_samples=args.mcs, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, verbose=-1, num_threads=10, seed=seed)
    m = lgb.train(params, lgb.Dataset(X[mask], yl, weight=w), args.iters)
    tmask = (meta._is_test == 1).values
    P = m.predict(X[tmask])
    ids = meta._txn_id[tmask].values
    order = np.argsort(ids)
    Ps.append((ids[order], P[order]))
    print('seed', seed, 'done', round(time.time() - t0), 's', flush=True)

ids = Ps[0][0]
P = np.mean([p for _, p in Ps], axis=0)
np.save(f'preds/{args.name}_test_P.npy', P); pd.Series(ids).to_csv(f'preds/{args.name}_test_ids.csv', index=False)

te = pd.read_csv('data/upi-fraud-rings/test.csv', usecols=['txn_id'])
meta0 = pd.read_parquet('/tmp/feat_s0.parquet', columns=['_month', '_is_test', '_txn_id'])
vids = meta0._txn_id[(meta0._is_test == 0) & (meta0._month == '2026-09')].values
vconn = pd.read_parquet('data/conn.parquet').set_index('txn_id').conn.reindex(vids).fillna(False).values.astype(bool)
for wm in args.wmode.split(','):
    wts = np.ones(len(LABELS))
    if args.val_P:
        Pv = np.load(args.val_P); yv = np.load('/tmp/y_val.npy'); assert len(Pv) == len(yv) == len(vids)
        keep = (yv == 0) | vconn if wm == 'proxy' else np.ones(len(yv), bool)
        wts, f = tune_weights(Pv[keep], yv[keep], iters=3)
        print(wm, 'val tuned F1', round(f, 4), 'weights', np.round(wts, 2).tolist())
    np.save(f'preds/{args.name}_{wm}_weights.npy', wts)
    pred = pd.Series(np.array(LABELS)[(P * wts).argmax(1)], index=ids)
    sub = pd.DataFrame({'id': te.txn_id, 'label': pred.reindex(te.txn_id).values})
    assert sub.label.notna().all() and len(sub) == len(te)
    sub.to_csv(f'submissions/{args.name}_{wm}.csv', index=False)
    print(wm, 'fraud share', round((sub.label != 'legit').mean(), 4), sub.label.value_counts().to_dict())
