"""E6: upper bound on what the residual (model-unflagged) Sep rows can give. Supervised, cross-fitted
inside Sep unflagged rows (51% frame): all v4 features + E2 entity-popularity/excess features.
Reports precision at the top-k rows and class purity there. If top-k precision stays near the 2.4%
base rate, no excess/density detector can do better (it has strictly less information)."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from load import load, R

d = load()
raw = pd.read_csv(R + 'data/upi-fraud-rings/train.csv', usecols=['timestamp', 'sender_id', 'receiver_id', 'device_id'])
ref = raw[raw.timestamp.str[:7].isin(['2026-07', '2026-08'])]
t = d[(d.grp == 'sep') & (~d.flag)].copy()
for k, col in [('snd', 'sender_id'), ('rcv', 'receiver_id'), ('dev', 'device_id')]:
    t[k + '_hist'] = t[col].map(ref[col].value_counts()).fillna(0).values
    t[k + '_cnt'] = t[col].map(t[col].value_counts()).values
DROP = ['ip_rank_month', 'cat_mismatch', 'app']
feats = [c for c in d.columns[:155] if c not in DROP and not c.startswith('_')] + [c for c in t.columns if c[:4] in ('snd_', 'rcv_', 'dev_')]
y = t.fraud.values
rng = np.random.default_rng(1); fold = rng.integers(0, 2, len(t)); s = np.zeros(len(t))
for k in range(2):
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.7, reg_lambda=5, verbose=-1, n_jobs=5)
    m.fit(t.loc[fold != k, feats], y[fold != k]); s[fold == k] = m.predict_proba(t.loc[fold == k, feats])[:, 1]
t['s'] = s
print('residual AUC fraud vs legit (Sep unflagged, x-fit):', round(roc_auc_score(y, s), 4), 'base', y.mean().round(4))
imp = pd.Series(m.booster_.feature_importance('gain'), index=feats).sort_values(ascending=False)
print((imp / imp.sum()).head(10).round(3).to_dict())
t = t.sort_values('s', ascending=False)
for k in [25, 50, 100, 200, 500, 1000, 2500]:
    top = t.head(k)
    vc = top[top.fraud == 1].y.value_counts()
    print(f'top {k:5d}: precision {top.fraud.mean():.3f}  best-class purity {vc.iloc[0] / k if len(vc) else 0:.3f} ({vc.index[0] if len(vc) else "-"})', vc.head(4).to_dict())
