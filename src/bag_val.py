"""Bagging check: train on Jul-Aug of sampling seeds S (iso7, drop cols), predict seed-0 Sep rows; average with seed-0 model.
usage: bag_val.py --seeds 1,2,3 --drop ip_rank_month --base /tmp/P_f3_noip_400.npy --out /tmp/P_bag.npy"""
import sys, argparse, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, 'src')
from features import cache_path, LABELS
from common import macro_f1, tune_weights
ap = argparse.ArgumentParser(); ap.add_argument('--seeds', default='1,2,3'); ap.add_argument('--drop', default='ip_rank_month')
ap.add_argument('--base', default=''); ap.add_argument('--out', default='/tmp/P_bag.npy'); ap.add_argument('--iters', type=int, default=400); ap.add_argument('--relabel', default='')
a = ap.parse_args()
STRUCT = [LABELS.index(c) for c in ['mule_account', 'velocity_bot', 'layering_chain', 'sim_swap', 'account_takeover', 'merchant_collusion', 'fake_cashback']]
conn_all = pd.read_parquet('data/conn_until_2026-09-01.parquet').set_index('txn_id').conn  # Jul-Aug labels only
drop = [c for c in a.drop.split(',') if c] + ['_label', '_month', '_is_test', '_txn_id', 'cat_mismatch']
X0 = pd.read_parquet(cache_path(0))
sep0 = ((X0._is_test == 0) & (X0._month == '2026-09')).values
y = X0._label[sep0].map({l: i for i, l in enumerate(LABELS)}).values.astype(int)
Xv = X0[sep0].drop(columns=[c for c in drop if c in X0.columns])
Ps = [np.load(a.base)] if a.base else []
for s in [int(x) for x in a.seeds.split(',')]:
    X = pd.read_parquet(cache_path(s))
    lab = X._label.map({l: i for i, l in enumerate(LABELS)}).fillna(-1).astype(int).values
    if a.relabel:
        rl = pd.read_parquet(a.relabel).set_index('txn_id').new_label.reindex(X._txn_id.values).values
        trmask = (X._is_test == 0).values & np.isin(X._month.values, ['2026-07', '2026-08'])
        lab = np.where(~np.isnan(rl) & trmask, rl, lab).astype(int)
    conn = conn_all.reindex(X._txn_id.values).fillna(False).values.astype(bool)
    bad = (lab > 0) & ~conn & np.isin(lab, STRUCT)
    trn = ((X._is_test == 0) & X._month.isin(['2026-07', '2026-08'])).values & ~bad
    rng = np.random.default_rng(s); mask = trn & ((lab != 0) | (rng.random(len(X)) < 0.25))
    Xt = X.drop(columns=[c for c in drop if c in X.columns])
    m = lgb.train(dict(objective='multiclass', num_class=12, learning_rate=0.05, num_leaves=63, min_child_samples=40, feature_fraction=0.7,
                       bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=10, seed=s),
                  lgb.Dataset(Xt[mask], lab[mask], weight=np.where(lab[mask] == 0, 4.0, 1.0)), a.iters)
    P = m.predict(Xv[Xt.columns]); Ps.append(P)
    print('seed', s, 'single F1 tuned', round(tune_weights(P, y, iters=2)[1], 4), flush=True)
P = np.mean(Ps, axis=0); np.save(a.out, P)
print('bag of', len(Ps), 'argmax', round(macro_f1(y, P.argmax(1)), 4), 'tuned', round(tune_weights(P, y, iters=3)[1], 4))
