"""Stage-2 ring model (collective classification, organiser hint: connect rows across accounts/phones/receivers/time).
Stage-1 class probabilities P (12) for every row of the seed-0 51% frame (train rows: week-blocked OOF /tmp/oof_s0.npy;
Sep validation rows: forward Jul-Aug model; test rows: full-train bag). Neighbour features = per entity (sender, receiver,
device, sender-as-receiver / receiver-as-sender money links) and window (1h, 24h, 7d): row count and summed P of the 11
fraud classes over OTHER rows. Stage-2 LGB on [P, neighbour feats, a few raw feats].
usage: ring2.py val  --valP /tmp/V_base_s0.npy               -> Sep x-fit of stage-1 vs stage-2, saves /tmp/V_ring2.npy
       ring2.py full --testP preds/<bag>_test_P.npy --tag ring2 -> preds/<tag>_test_P.npy (+ids)"""
import sys, argparse, time
import numpy as np, pandas as pd
sys.path.insert(0, 'src')
from features import load, sample_frame, LABELS
from common import xfit_f1

ap = argparse.ArgumentParser(); ap.add_argument('mode'); ap.add_argument('--valP', default='/tmp/V_base_s0.npy')
ap.add_argument('--testP', default=''); ap.add_argument('--tag', default='ring2'); ap.add_argument('--threads', type=int, default=8)
ap.add_argument('--iters', type=int, default=300)
a = ap.parse_args()
L2I = {l: i for i, l in enumerate(LABELS)}
WIN = {'1h': 3600, '24h': 86400, '7d': 7 * 86400}

tr, te, _, _ = load(); df = sample_frame(tr, te, 0)
ts = (pd.to_datetime(df.timestamp) - pd.Timestamp('2026-06-01')).dt.total_seconds().values.astype(np.int64)
mon = df.timestamp.str[:7].values; ist = (df._is_test == 1).values
lab = df.label.map(L2I).fillna(-1).astype(int).values
P = np.load('/tmp/oof_s0.npy').copy()
if a.mode == 'val':
    V = np.load(a.valP); vid = np.load(a.valP.replace('.npy', '_ids.npy'), allow_pickle=True)
    pos = pd.Series(np.arange(len(df)), index=df.txn_id.values).reindex(vid).values; P[pos] = V
    P[ist] = np.nan  # Oct rows unused in validation
else:
    T = np.load(a.testP); tid = pd.read_csv(a.testP.replace('_test_P.npy', '_test_ids.csv')).iloc[:, 0].values
    pos = pd.Series(np.arange(len(df)), index=df.txn_id.values).reindex(tid).values; P[pos] = T
P = np.nan_to_num(P)
F = P[:, 1:]  # fraud-class probs


def nb_sums(key_a, key_b, W):
    """For each row i: sum of F over rows j != i with key_b[j] == key_a[i] and |ts_j - ts_i| <= W."""
    kb = pd.Series(key_b).astype('category'); cats = kb.cat.categories
    ka = pd.Categorical(key_a, categories=cats).codes.astype(np.int64); kbc = kb.cat.codes.values.astype(np.int64)
    big = np.int64(10 ** 9)
    o = np.lexsort((ts, kbc)); sk = kbc[o] * big + ts[o]
    cs = np.vstack([np.zeros((1, F.shape[1] + 1)), np.cumsum(np.hstack([F[o], np.ones((len(o), 1))]), 0)])
    q = np.where(ka >= 0, ka, -5) * big
    lo = np.searchsorted(sk, q + ts - W, 'left'); hi = np.searchsorted(sk, q + ts + W, 'right')
    s = cs[hi] - cs[lo]
    same = (key_a == key_b)  # row itself counted when its own key matches
    s[same] -= np.hstack([F[same], np.ones((same.sum(), 1))])
    s[ka < 0] = 0
    return s


t0 = time.time(); feats = {}
snd, rcv, dev = df.sender_id.values, df.receiver_id.values, df.device_id.values
for nm, ka, kb in [('s', snd, snd), ('r', rcv, rcv), ('d', dev, dev), ('sr', snd, rcv), ('rs', rcv, snd)]:
    for wn, W in WIN.items():
        s = nb_sums(ka, kb, W)
        feats[f'{nm}{wn}_n'] = s[:, -1]; feats[f'{nm}{wn}_pf'] = s[:, :-1].sum(1)
        for c in range(11):
            feats[f'{nm}{wn}_{LABELS[c + 1][:5]}'] = s[:, c]
X = pd.DataFrame(feats)
X[[f'p_{l[:6]}' for l in LABELS]] = P
X['amount'] = df.amount.values; X['p2p'] = (df.receiver_type.values == 'P2P').astype(int)
print('feats', X.shape, f'{time.time()-t0:.0f}s', flush=True)

import lightgbm as lgb
prm = dict(objective='multiclass', num_class=12, learning_rate=0.05, num_leaves=31, min_child_samples=40, feature_fraction=0.7,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=a.threads, seed=0)
rng = np.random.default_rng(0)
bad = np.isin(df.txn_id.values, pd.read_parquet('data/oofbadfull_u.parquet').txn_id.values)
if a.mode == 'val':
    trn = ~ist & np.isin(mon, ['2026-07', '2026-08']) & ~bad & ((lab > 0) | ((lab == 0) & (rng.random(len(df)) < 0.25)))
    m = lgb.train(prm, lgb.Dataset(X[trn], lab[trn], weight=np.where(lab[trn] == 0, 4.0, 1.0)), a.iters)
    sep = pos
    y = lab[sep]; P2 = m.predict(X.iloc[sep])
    np.save('/tmp/V_ring2.npy', P2); np.save('/tmp/V_ring2_ids.npy', vid)
    s1 = [xfit_f1(P[sep], y, seed=k) for k in range(5)]; s2 = [xfit_f1(P2, y, seed=k) for k in range(5)]
    print('Sep x-fit stage1', round(np.mean(s1), 4), 'stage2', round(np.mean(s2), 4), f'{time.time()-t0:.0f}s')
    imp = pd.Series(m.feature_importance('gain'), index=X.columns).sort_values(ascending=False); print(imp.head(20).round(0))
else:
    trn = ~ist & (lab >= 0) & ~bad & ((lab > 0) | ((lab == 0) & (rng.random(len(df)) < 0.25)))
    m = lgb.train(prm, lgb.Dataset(X[trn], lab[trn], weight=np.where(lab[trn] == 0, 4.0, 1.0)), a.iters)
    P2 = m.predict(X.iloc[pos]); np.save(f'preds/{a.tag}_test_P.npy', P2)
    pd.Series(tid).to_csv(f'preds/{a.tag}_test_ids.csv', index=False); print('saved', a.tag, f'{time.time()-t0:.0f}s')
