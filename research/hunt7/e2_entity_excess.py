"""E2: entity-level excess. Under 'separate injection process', injected rows pick entities with a different
(e.g. flatter) distribution than legit, so entities would get MORE rows in the target month than their history predicts.
For each target row: history popularity (reference months, full density) and target-month excess
(count in target month on the 51% frame, unflagged rows only, minus expected from history).
backward: ref = Jul-Aug, target = Sep (labels -> AUC/lift).  forward: ref = Jul-Sep, target = Oct."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from load import load, R, STRUCT

d = load()
raw = pd.concat([pd.read_csv(R + 'data/upi-fraud-rings/train.csv', usecols=['txn_id', 'timestamp', 'sender_id', 'receiver_id', 'device_id', 'label']),
                 pd.read_csv(R + 'data/upi-fraud-rings/test.csv', usecols=['txn_id', 'timestamp', 'sender_id', 'receiver_id', 'device_id'])])
raw['m'] = raw.timestamp.str[:7]


def feats(ref_months, tgt_grp):
    ref = raw[raw.m.isin(ref_months)]
    nref = len(ref)
    t = d[(d.grp == tgt_grp) & (~d.flag)].copy()
    ntgt = len(d[d.grp == tgt_grp])
    out = {}
    for k, col in [('snd', 'sender_id'), ('rcv', 'receiver_id'), ('dev', 'device_id')]:
        h = ref[col].value_counts()
        t[k + '_hist'] = t[col].map(h).fillna(0).values / nref * 1e6
        c = t[col].map(t[col].value_counts()).values  # count in target month among unflagged rows (51% frame)
        exp = t[k + '_hist'] / 1e6 * ntgt
        t[k + '_cnt'] = c
        t[k + '_exc'] = (c - exp) / np.sqrt(exp + 1)
        t[k + '_new'] = (t[k + '_hist'] == 0).astype(int)
    # pair history
    ph = ref.groupby(['sender_id', 'receiver_id']).size()
    t['pair_hist'] = pd.MultiIndex.from_frame(t[['sender_id', 'receiver_id']]).map(lambda x: 0).values if False else \
        t.set_index(['sender_id', 'receiver_id']).index.map(ph).fillna(0).values
    return t


cols = lambda t: [c for c in t.columns if c.split('_')[0] in ('snd', 'rcv', 'dev', 'pair') and c.split('_')[-1] in ('hist', 'cnt', 'exc', 'new')]
t = feats(['2026-07', '2026-08'], 'sep')
t['iso'] = (t.fraud == 1) & (t.conn == False)
F = cols(t)
rows = []
for cls in [None] + STRUCT + ['phishing_collect', 'qr_swap', 'refund_abuse', 'fake_cashback', 'loan_app_extortion']:
    for rt in ['MERCHANT', 'P2P']:
        s = t[(t.rt == rt)]
        s = s[(s.y == 'legit') | ((s.fraud == 1) if cls is None else (s.y == cls))]
        y = (s.y != 'legit').values
        if y.sum() < 30: continue
        r = {'cls': cls or 'ALL', 'rt': rt, 'n': int(y.sum())}
        for f in F:
            a = roc_auc_score(y, s[f].values)
            r[f] = round(a, 3)
        rows.append(r)
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 40)
res = pd.DataFrame(rows)
print(res.to_string())
# best single-direction lift cells
for rt in ['MERCHANT', 'P2P']:
    s = t[t.rt == rt]
    for f in ['rcv_new', 'snd_new', 'dev_new']:
        g = s.groupby(f).fraud.agg(['mean', 'size'])
        print(rt, f, g.round(4).to_dict('index'))
    for f in ['rcv_hist', 'rcv_exc', 'snd_hist', 'snd_exc']:
        q = pd.qcut(s[f].rank(method='first'), 10, labels=False)
        print(rt, f, 'decile fraud rate', s.groupby(q).fraud.mean().round(4).tolist())
