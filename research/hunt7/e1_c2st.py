"""E1: classifier two-sample test (excess detector), label-free.
backward: reference = Jul-Aug legit-labelled rows (51% frame), target = Sep rows v6 does not flag.
          Cross-fitted score on Sep target rows -> AUC of labelled fraud vs legit (does excess = legit-profile fraud?).
forward:  reference = Sep legit-labelled, target = Oct rows v11 does not flag. Inspect Oct-leaning rows.
Drift-prone columns (absolute amounts/balances, ages that grow with calendar time, IP, app) are dropped."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from load import load, STRUCT

mode = sys.argv[1] if len(sys.argv) > 1 else 'backward'
d = load()
DROP = ['amount', 'log_amount', 'balance_before', 'bal_after', 'sender_account_age_days', 'recv_acc_age_days',
        'merchant_age_days', 'ip_rank_month', 'cat_mismatch', 'weekday', 'app', 's_n_p30d_amt', 's_n_p7d_amt', 's_n_1d_amt',
        's_n_1h_amt', 'r_n_1h_amt', 's_in_prev1h_amt', 'rev_amt', 'rev_next_amt']
feats = [c for c in d.columns[:155] if c not in DROP and not c.startswith('_')]
import re
if 'short' in sys.argv: feats = [c for c in feats if not re.search(r'p7d|p30d|30d|7d|3d|_share|r_1d_vs|amt_vs_|p_share|s_in_p30d|s_last_in|ratio', c)]
print(len(feats), feats)
if mode == 'backward':
    ref = d[(d._month == '2026-08') & (d.y == 'legit')]
    tgt = d[(d.grp == 'sep') & (~d.flag)]
else:
    ref = d[(d.grp == 'sep') & (d.y == 'legit')]
    tgt = d[(d.grp == 'oct') & (~d.flag)]
x = pd.concat([ref.assign(z=0), tgt.assign(z=1)], ignore_index=True)
for c in ['receiver_type', 'txn_type', 'merchant_category', 'sender_os']:
    if c in feats: x[c] = x[c].astype('category')
rng = np.random.default_rng(0)
fold = rng.integers(0, 2, len(x))
score = np.zeros(len(x))
for k in range(2):
    tr, va = fold != k, fold == k
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=100, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=5)
    m.fit(x.loc[tr, feats], x.z[tr])
    score[va] = m.predict_proba(x.loc[va, feats])[:, 1]
x['s'] = score
print(mode, 'C2ST AUC ref vs target', round(roc_auc_score(x.z, score), 4))
imp = pd.Series(m.booster_.feature_importance('gain'), index=feats).sort_values(ascending=False)
print('top features', (imp / imp.sum()).head(12).round(3).to_dict())
t = x[x.z == 1].copy()
if mode == 'backward':
    y = t.fraud.values
    print('AUC (Sep unflagged) fraud vs legit:', round(roc_auc_score(y, t.s), 4), 'base rate', y.mean().round(4))
    iso = t[(t.y == 'legit') | (t.y.isin(STRUCT) & (t.conn == False))]
    print('AUC isolated struct vs legit:', round(roc_auc_score(iso.fraud, iso.s), 4))
    for cls in ['mule_account', 'velocity_bot', 'layering_chain', 'sim_swap', 'account_takeover', 'phishing_collect', 'qr_swap',
                'refund_abuse', 'fake_cashback', 'loan_app_extortion', 'merchant_collusion']:
        s = t[t.y.isin(['legit', cls])]
        print(f'  {cls:20s} n={int((s.y == cls).sum()):5d} AUC={roc_auc_score(s.y == cls, s.s):.3f}')
    for q in [0.99, 0.975, 0.95, 0.9]:
        th = np.quantile(t.s, q); top = t[t.s >= th]
        print(f'top {1 - q:.3f}: fraud rate {top.fraud.mean():.4f} lift {top.fraud.mean() / y.mean():.2f}')
else:
    for q in [0.99, 0.975, 0.95]:
        th = np.quantile(x.s[x.z == 1], q)
        print(f'q{q}: oct above {(t.s >= th).sum()} sep-legit above {(x[(x.z == 0)].s >= th).sum()} (scaled x{len(t) / (x.z == 0).sum():.3f})')
    t[['txn_id', 's']].to_parquet(f'/tmp/hunt7/e1_{mode}.parquet')
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print(t.sort_values('s', ascending=False)[['txn_type', 'receiver_type', 'merchant_category', 'hour', 'amount', 'amt_over_bal',
          'new_payee', 'sim_age_hours', 'device_age_days', 'device_is_own', 'login_to_txn_sec', 's_n_1d', 'r_n_1d', 's']].head(25))
