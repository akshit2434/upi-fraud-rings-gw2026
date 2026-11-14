"""E3: tiny linked structures (pairs/groups of rows) per 10k rows: Oct vs Sep (51% frame) vs Sep (full density).
Excess in Oct among rows the model does NOT flag would point at a separate fraud process.
Backward check: fraud rate of Sep-unflagged rows inside each structure (51% frame)."""
import numpy as np, pandas as pd
from load import load, R

d = load()
raw = pd.read_csv(R + 'data/upi-fraud-rings/train.csv')
raw['t'] = (pd.to_datetime(raw.timestamp) - pd.Timestamp('2026-07-01')).dt.total_seconds()
sepfull = raw[raw.timestamp.str[:7] == '2026-09'].copy()
sepfull['flag'] = False  # no model flags at full density; report all rows + legit-only
sepfull['y'] = sepfull.label
cols = ['txn_id', 't', 'sender_id', 'receiver_id', 'device_id', 'amount', 'rt', 'tt', 'flag', 'y', 'dnum']
sep = d[d.grp == 'sep'][cols].copy()
oct_ = d[d.grp == 'oct'][cols].copy()
sepfull = sepfull.rename(columns={'receiver_type': 'rt', 'txn_type': 'tt'})
sepfull['dnum'] = sepfull.device_id.str[2:].astype(int)
sepfull = sepfull[cols]


def link_count(x, key, win, extra=None):
    """number of OTHER rows sharing key (and extra cols) within +-win seconds."""
    k = [key] + (extra or [])
    x = x.sort_values(k + ['t'])
    g = x.groupby(k, sort=False).t
    prev = x.t - g.shift(1); nxt = g.shift(-1) - x.t
    prev2 = x.t - g.shift(2); nxt2 = g.shift(-2) - x.t
    n = (prev <= win).astype(int) + (nxt <= win).astype(int) + (prev2 <= win).astype(int) + (nxt2 <= win).astype(int)
    return n.reindex(x.index)


def hop(x, win):
    """receiver (P2P) sends money on within win seconds after this row."""
    s = x[['sender_id', 't']].sort_values(['sender_id', 't'])
    keys = s.sender_id.values; ts = s.t.values
    r = x.receiver_id.values; tt = x.t.values
    # searchsorted on combined key
    order = np.lexsort((ts, keys)); keys, ts = keys[order], ts[order]
    lo = np.searchsorted(keys, r, 'left'); hi = np.searchsorted(keys, r, 'right')
    out = np.zeros(len(x), bool)
    for i in np.where(hi > lo)[0]:
        seg = ts[lo[i]:hi[i]]
        j = np.searchsorted(seg, tt[i], 'right')
        out[i] = j < len(seg) and seg[j] - tt[i] <= win
    return out


def structs(x):
    x = x.copy()
    p2p = x.rt == 'P2P'
    S = {}
    S['snd_10m'] = link_count(x, 'sender_id', 600) > 0
    S['snd_1h'] = link_count(x, 'sender_id', 3600) > 0
    S['snd_sameamt_1d'] = link_count(x, 'sender_id', 86400, ['amount']) > 0
    S['rcvP2P_1h'] = p2p & (link_count(x, 'receiver_id', 3600) > 0)
    S['rcvP2P_sameamt_1d'] = p2p & (link_count(x, 'receiver_id', 86400, ['amount']) > 0)
    S['rcvM_sameamt_1h'] = (~p2p) & (link_count(x, 'receiver_id', 3600, ['amount']) > 0)
    S['pair_1d'] = link_count(x, 'sender_id', 86400, ['receiver_id']) > 0
    S['dev_nonown_1h'] = (x.dnum >= 420000) & (link_count(x, 'device_id', 3600) > 0)
    S['hop_1h'] = p2p & hop(x, 3600)
    return pd.DataFrame(S, index=x.index)


rows = {}
for name, x in [('sep51', sep), ('sepfull', sepfull), ('oct', oct_)]:
    S = structs(x)
    n = len(x)
    leg = (x.y == 'legit') if name != 'oct' else pd.Series(True, index=x.index)
    unf = ~x.flag.astype(bool)
    r = {}
    for c in S:
        r[(c, 'all/10k')] = S[c].sum() / n * 1e4
        r[(c, 'unflag/10k')] = (S[c] & unf).sum() / n * 1e4
        if name != 'oct':
            r[(c, 'legit/10k')] = (S[c] & leg).sum() / n * 1e4
            sub = x[S[c] & unf]
            r[(c, 'unflag_fraud%')] = 100 * (sub.y != 'legit').mean()
    rows[name] = r
    if name == 'sep51':
        print('sep51 unflagged base fraud%', 100 * (x[unf].y != 'legit').mean())
out = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(out.round(2).to_string())
