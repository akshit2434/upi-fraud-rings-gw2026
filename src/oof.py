"""OOF class probabilities for every TRAIN row of a sampling-seed frame (week-blocked folds, iso7-cleaned training, drop ip).
Saves /tmp/oof_s<seed>.npy aligned with the cached frame rows (test rows = NaN)."""
import sys, numpy as np, pandas as pd, lightgbm as lgb, time
sys.path.insert(0, 'src')
from features import cache_path, LABELS
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
K = 4
X = pd.read_parquet(cache_path(seed))
meta = X[['_label', '_month', '_is_test', '_txn_id']]
X = X.drop(columns=['_label', '_month', '_is_test', '_txn_id', 'cat_mismatch', 'ip_rank_month'], errors='ignore')
lab = meta._label.map({l: i for i, l in enumerate(LABELS)}).fillna(-1).astype(int).values
conn = pd.read_parquet('data/conn.parquet').set_index('txn_id').conn.reindex(meta._txn_id.values).fillna(False).values.astype(bool)
STRUCT = [LABELS.index(c) for c in ['mule_account', 'velocity_bot', 'layering_chain', 'sim_swap', 'account_takeover', 'merchant_collusion', 'fake_cashback']]
bad = (lab > 0) & ~conn & np.isin(lab, STRUCT)
tr = (meta._is_test == 0).values
# week index from the time features (minute-of-day not enough) -> use txn timestamp via df
from features import load, sample_frame
trd, te, acc, mer = load(); df = sample_frame(trd, te, seed); assert (df.txn_id.values == meta._txn_id.values).all()
week = ((pd.to_datetime(df.timestamp) - pd.Timestamp('2026-07-01')).dt.days.values // 7)
fold = week % K
OOF = np.full((len(X), 12), np.nan)
rng = np.random.default_rng(seed)
keep_legit = rng.random(len(X)) < 0.25
for f in range(K):
    t0 = time.time()
    fit = tr & ~bad & (fold != f) & ((lab != 0) | keep_legit)
    m = lgb.train(dict(objective='multiclass', num_class=12, learning_rate=0.05, num_leaves=63, min_child_samples=40, feature_fraction=0.7,
                       bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=10, seed=seed + f),
                  lgb.Dataset(X[fit], lab[fit], weight=np.where(lab[fit] == 0, 4.0, 1.0)), 400)
    sel = tr & (fold == f)
    OOF[sel] = m.predict(X[sel])
    print('fold', f, round(time.time() - t0), 's', flush=True)
np.save(f'/tmp/oof_s{seed}.npy', OOF)
print('saved')
