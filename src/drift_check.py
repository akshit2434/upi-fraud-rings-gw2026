"""Does a feature keep its fraud signal in October? Train WITHOUT the probed features (Jul-Aug, iso7-clean),
score Sep (labelled) and Oct test, take the top-K most-fraud-like rows per predicted class, and compare the probed
features' means there vs among low-risk rows, Sep vs Oct."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, 'src')
from features import cache_path, LABELS
PROBE = ['ip_rank_month', 'qr_vpa_mismatch', 'qr_checked', 'new_payee', 'pin_retries', 'login_to_txn_sec', 'app', 'device_age_days', 'sim_age_hours', 'away_from_home', 'amt_rank_month']
X = pd.read_parquet(cache_path(0))
meta = X[['_label', '_month', '_is_test', '_txn_id']]; X = X.drop(columns=['_label', '_month', '_is_test', '_txn_id', 'cat_mismatch'], errors='ignore')
lab = meta._label.map({l: i for i, l in enumerate(LABELS)})
conn = pd.read_parquet('data/conn.parquet').set_index('txn_id').conn.reindex(meta._txn_id.values).fillna(False).values.astype(bool)
STRUCT = [LABELS.index(c) for c in ['mule_account', 'velocity_bot', 'layering_chain', 'sim_swap', 'account_takeover', 'merchant_collusion', 'fake_cashback']]
bad = (lab.fillna(0).values > 0) & ~conn & np.isin(lab.fillna(0).values, STRUCT)
trn = (meta._is_test == 0) & meta._month.isin(['2026-07', '2026-08']) & ~bad
rng = np.random.default_rng(0); mask = trn & ((lab != 0) | (rng.random(len(X)) < 0.25))
Xn = X.drop(columns=PROBE)
m = lgb.train(dict(objective='multiclass', num_class=12, learning_rate=0.05, num_leaves=63, min_child_samples=40, feature_fraction=0.7,
                   bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=10, seed=0), lgb.Dataset(Xn[mask], lab[mask].values,
                   weight=np.where(lab[mask].values == 0, 4.0, 1.0)), 300)
sep = ((meta._is_test == 0) & (meta._month == '2026-09')).values
octm = ((meta._is_test == 1) & (meta._month == '2026-10')).values
out = []
for nm, sel in [('Sep', sep), ('Oct', octm)]:
    P = m.predict(Xn[sel]); fr = 1 - P[:, 0]
    k = int(0.006 * sel.sum())
    top = np.argsort(-fr)[:k]; low = np.argsort(fr)[: sel.sum() // 2]
    Xs = X[sel].reset_index(drop=True)
    r = {'set': nm, 'top_frac_true_fraud': (lab[sel].values[top] > 0).mean() if nm == 'Sep' else np.nan}
    for c in PROBE:
        r[c] = f'{Xs[c].iloc[top].mean():.3f} vs {Xs[c].iloc[low].mean():.3f}'
    out.append(r)
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 30)
print(pd.DataFrame(out).T.to_string())
