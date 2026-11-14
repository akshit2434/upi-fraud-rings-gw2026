"""F5: multi-hop money chains among UNFLAGGED P2P rows (row i -> row j if j.sender == i.receiver and 0 < t_j - t_i <= W).
Count rows in chains of depth >=2 / >=3 per 10k, Oct (v12) vs Sep51 (v6); Sep fraud share in them. Also fan-out: sender with
>=3 distinct P2P receivers within 6h, and fan-in: P2P receiver with >=3 distinct senders within 24h (unflagged rows)."""
import numpy as np, pandas as pd
from load import load, R
d = load()
v12 = pd.read_csv(R + 'submissions/v12_tailfix.csv').rename(columns={'id': 'txn_id', 'label': 'v12'})
d = d.merge(v12, on='txn_id', how='left'); d['flag'] = np.where(d.grp == 'oct', d.v12 != 'legit', d.flag)
out = {}
for g in ['sep', 'oct']:
    x = d[(d.grp == g) & (d.rt == 'P2P') & (~d.flag.astype(bool))][['txn_id', 'sender_id', 'receiver_id', 't', 'fraud', 'y', 'amount']].reset_index(drop=True)
    n_all = (d.grp == g).sum()
    for W in [3600, 6 * 3600, 86400]:
        a = x[['receiver_id', 't', 'amount']].rename(columns={'receiver_id': 'k'}).reset_index().rename(columns={'index': 'i'})
        b = x[['sender_id', 't', 'amount']].rename(columns={'sender_id': 'k'}).reset_index().rename(columns={'index': 'j'})
        e = a.merge(b, on='k', suffixes=('_i', '_j'))
        e = e[(e.t_j > e.t_i) & (e.t_j - e.t_i <= W)]
        has_out = np.zeros(len(x), bool); has_in = np.zeros(len(x), bool)
        has_out[e.i.values] = True; has_in[e.j.values] = True
        mid = has_out & has_in  # row in the middle of a >=3-row chain
        member = has_out | has_in
        less = e[(e.amount_j < e.amount_i) & (e.amount_j > 0.8 * e.amount_i)]  # 'a little less each hop'
        lm = np.zeros(len(x), bool); lm[less.i.values] = True; lm[less.j.values] = True
        r = {'chain_member': member.sum(), 'chain_mid(>=3)': mid.sum(), 'hop_80-100%amt': lm.sum()}
        for k, m in r.items():
            out[(g, W, k)] = {'rows': int(m), 'per10k': round(m / n_all * 1e4, 2), 'fraud%': round(100 * x.fraud.values[{'chain_member': member, 'chain_mid(>=3)': mid, 'hop_80-100%amt': lm}[k]].mean(), 1) if g == 'sep' else None}
    xs = x.sort_values('t')
    fo = xs.groupby('sender_id').receiver_id.transform('nunique'); out[(g, 'all', 'sender>=3 P2P recv in month')] = {'rows': int((fo >= 3).sum()), 'per10k': round((fo >= 3).sum() / n_all * 1e4, 2), 'fraud%': round(100 * xs.fraud[fo >= 3].mean(), 1) if g == 'sep' else None}
    fi = xs.groupby('receiver_id').sender_id.transform('nunique'); out[(g, 'all', 'recv>=4 P2P senders in month')] = {'rows': int((fi >= 4).sum()), 'per10k': round((fi >= 4).sum() / n_all * 1e4, 2), 'fraud%': round(100 * xs.fraud[fi >= 4].mean(), 1) if g == 'sep' else None}
pd.set_option('display.width', 200)
print(pd.DataFrame(out).T.to_string())
