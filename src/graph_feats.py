"""Graph / event features on the 51%-density frame (same rows & order as features.load_cached(seed)).
Rows are linked into short-lived 'events' (union-find):
  same device gap <= 30 min · same P2P receiver gap <= 60 min · same sender gap <= 60 min · money hop
  (row i pays r, r's next outgoing payment within 60 min).
Per row: component size / distinct entities / duration / composition, plus multi-hop chain depth forward & backward.
All windows are short (<= 1h links), so no edge drift at the test-period boundary beyond what 1h windows have.
usage: graph_feats.py --seeds 0,1,2,3  -> /tmp/graph_<FEATURE_VERSION>_s<seed>.parquet"""
import sys, time, argparse
import numpy as np, pandas as pd
from numba import njit
sys.path.insert(0, 'src')
from features import load, sample_frame, FEATURE_VERSION, T0

GAP = {'device_id': 1800, 'receiver_p2p': 3600, 'sender_id': 3600}
HOP = 3600


@njit(cache=True)
def _find(p, i):
    while p[i] != i:
        p[i] = p[p[i]]
        i = p[i]
    return i


@njit(cache=True)
def _union_all(p, a, b):
    for k in range(len(a)):
        x = _find(p, a[k]); y = _find(p, b[k])
        if x != y:
            if x < y:
                p[y] = x
            else:
                p[x] = y
    for i in range(len(p)):
        p[i] = _find(p, i)


def consecutive_links(key, t, gap, valid=None):
    """Link consecutive rows of the same key when the time gap <= gap."""
    idx = np.arange(len(key)) if valid is None else np.flatnonzero(valid)
    o = idx[np.lexsort((t[idx], key[idx]))]
    same = (key[o][1:] == key[o][:-1]) & (np.diff(t[o]) <= gap)
    return o[:-1][same], o[1:][same]


def hop_next(df, t):
    """For row i (s->r, r is P2P), index of r's first outgoing row j with t_i <= t_j <= t_i + HOP, else -1."""
    n = len(df)
    q = pd.DataFrame({'k': df.receiver_id.values, 't': t, 'i': np.arange(n)})[df.receiver_type.values == 'P2P']
    s = pd.DataFrame({'k': df.sender_id.values, 't': t, 'j': np.arange(n)})
    q = q.sort_values('t'); s = s.sort_values('t')
    m = pd.merge_asof(q, s, on='t', by='k', direction='forward', tolerance=HOP, allow_exact_matches=True)
    out = np.full(n, -1, np.int64)
    ok = m.j.notna().values
    out[m.i.values[ok]] = m.j.values[ok].astype(np.int64)
    out[out == np.arange(n)] = -1
    return out


@njit(cache=True)
def _depth(nxt, iters):
    d = np.zeros(len(nxt), np.int64)
    for _ in range(iters):
        changed = False
        for i in range(len(nxt)):
            j = nxt[i]
            if j >= 0:
                v = d[j] + 1
                if v > d[i] and v < 50:
                    d[i] = v; changed = True
        if not changed:
            break
    return d


def build(df):
    n = len(df)
    t = (pd.to_datetime(df.timestamp) - T0).dt.total_seconds().values.astype(np.int64)
    codes = {c: pd.factorize(df[c])[0] for c in ['sender_id', 'receiver_id', 'device_id']}
    p2p = df.receiver_type.values == 'P2P'
    A, B = [], []
    for k, g in [('device_id', GAP['device_id']), ('sender_id', GAP['sender_id'])]:
        a, b = consecutive_links(codes[k], t, g); A.append(a); B.append(b)
    a, b = consecutive_links(codes['receiver_id'], t, GAP['receiver_p2p'], valid=p2p); A.append(a); B.append(b)
    nxt = hop_next(df, t)
    ok = nxt >= 0; A.append(np.flatnonzero(ok)); B.append(nxt[ok])
    A = np.concatenate(A).astype(np.int64); B = np.concatenate(B).astype(np.int64)
    par = np.arange(n, dtype=np.int64)
    _union_all(par, A, B)
    comp = par
    amt = df.amount.values
    own = (df.device_id.str[3:].values == df.sender_id.str[2:].values)
    hour = pd.to_datetime(df.timestamp).dt.hour.values
    c = pd.DataFrame({'c': comp, 't': t, 's': codes['sender_id'], 'r': codes['receiver_id'], 'd': codes['device_id'],
                      'amt': amt, 'la': np.log1p(amt), 'p2p': p2p.astype(float), 'newp': df.new_payee.values.astype(float),
                      'notown': (~own).astype(float), 'fsim': (df.sim_age_hours.values < 72).astype(float),
                      'newdev': (df.device_age_days.values < 2).astype(float), 'night': ((hour <= 5) | (hour >= 23)).astype(float),
                      'pin': (df.pin_retries.values > 0).astype(float), 'hop': ok.astype(float),
                      'round': (np.mod(amt, 100) == 0).astype(float),
                      'col': (df.txn_type.values == 'COLLECT').astype(float), 'qr': (df.txn_type.values == 'QR').astype(float)})
    g = c.groupby('c')
    F = pd.DataFrame(index=np.arange(n))
    F['g_size'] = g.t.transform('size').values
    F['g_nsend'] = g.s.transform('nunique').values
    F['g_nrecv'] = g.r.transform('nunique').values
    F['g_ndev'] = g.d.transform('nunique').values
    F['g_dur'] = (g.t.transform('max') - g.t.transform('min')).values
    F['g_amt_sum'] = g.amt.transform('sum').values
    F['g_la_std'] = g.la.transform('std').fillna(0).values
    for k in ['p2p', 'newp', 'notown', 'fsim', 'newdev', 'night', 'pin', 'hop', 'round', 'col', 'qr']:
        F[f'g_{k}'] = g[k].transform('mean').values
    F['g_rank_t'] = g.t.rank(method='first').values / F['g_size'].values
    F['g_dev_per_send'] = F['g_ndev'] / F['g_nsend']
    F['g_send_per_recv'] = F['g_nsend'] / F['g_nrecv']
    # chain depth: forward (how many hops the money keeps moving) and backward (how many hops brought it here)
    F['hop_fwd'] = _depth(nxt, 30)
    prv = np.full(n, -1, np.int64); prv[nxt[ok]] = np.flatnonzero(ok)
    F['hop_bwd'] = _depth(prv, 30)
    F['hop_ratio'] = np.where(ok, amt[np.maximum(nxt, 0)] / amt, np.nan)
    F['hop_dt'] = np.where(ok, t[np.maximum(nxt, 0)] - t, np.nan)
    F['hop_dec'] = np.where(ok, ((F['hop_ratio'] > 0.8) & (F['hop_ratio'] <= 1.0)).astype(float), np.nan)
    # out-degree fan: how many distinct receivers the receiver pays within 1h after (fan-out of mule)
    return F.astype(np.float32)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', default='0'); a = ap.parse_args()
    tr, te, acc, mer = load()
    for s in [int(x) for x in a.seeds.split(',')]:
        t0 = time.time()
        df = sample_frame(tr, te, s).reset_index(drop=True)
        F = build(df); F['_txn_id'] = df.txn_id.values
        F.to_parquet(f'/tmp/graph_{FEATURE_VERSION}_s{s}.parquet')
        print('seed', s, F.shape, 'giant comp', int(F.g_size.max()), f'{time.time()-t0:.0f}s', flush=True)
