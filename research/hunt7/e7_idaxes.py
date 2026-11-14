"""E7: allocation-axis residual scan. On Sep rows v6 does NOT flag, can raw ID numbers / raw columns
(device number, sender/receiver number, txn_id hex, timestamp, ip, app_version as DIAGNOSTIC only) find blocks
of residual fraud? Positive control: device number (the known fraud block). x-fit LGB, one axis set at a time."""
import numpy as np, pandas as pd, lightgbm as lgb, sys
from sklearn.metrics import roc_auc_score
from load import load, R

d = load()
tr = pd.read_csv(R + 'data/upi-fraud-rings/train.csv', usecols=['txn_id', 'app_version', 'amount', 'balance_before'])
t = d[(d.grp == 'sep') & (~d.flag)].merge(tr, on='txn_id', how='left', suffixes=('', '_raw'))
t['snum'] = t.sender_id.str[2:].astype(int)
t['rnum'] = t.receiver_id.str[2:].astype(int); t['r_is_m'] = (t.rt == 'MERCHANT').astype(int)
t['txhex'] = t.txn_id.str[2:].apply(lambda s: int(s, 16)).astype(float)
t['sec'] = t.t % 60; t['tmod'] = t.t
t['ver'] = t.app_version.astype(float)
t['amt_p'] = (np.round(t.amount_raw * 100) % 100); t['bal_p'] = (np.round(t.balance_before_raw * 100) % 100)
y = t.fraud.values
sets = {'device_num': ['dnum'], 'sender_num': ['snum'], 'receiver_num': ['rnum', 'r_is_m'], 'txn_hex': ['txhex'],
        'timestamp': ['tmod', 'sec'], 'ip(diag)': ['ip_risk_score'], 'app_ver(diag)': ['ver'], 'paise': ['amt_p', 'bal_p'],
        'all_ids': ['dnum', 'snum', 'rnum', 'r_is_m', 'txhex', 'tmod']}
for name, f in sets.items():
    rng = np.random.default_rng(2); fold = rng.integers(0, 2, len(t)); s = np.zeros(len(t))
    for k in range(2):
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=50, verbose=-1, n_jobs=5)
        m.fit(t.loc[fold != k, f], y[fold != k]); s[fold == k] = m.predict_proba(t.loc[fold == k, f])[:, 1]
    o = np.argsort(-s)
    p = {k: round(y[o[:k]].mean(), 3) for k in [50, 100, 200, 500]}
    print(f'{name:14s} AUC {roc_auc_score(y, s):.4f} top-k precision {p}')
