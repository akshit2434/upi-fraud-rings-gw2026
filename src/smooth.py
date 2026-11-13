"""Event-level smoothing: average class probs over graph components (graph_feats link rules) on a frame.
comp_ids(seed) -> component id per row of the frame (same order as load_cached(seed))."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'src')
import graph_feats as gf
from features import load, sample_frame, T0


def comp_ids(seed, tr=None, te=None):
    if tr is None:
        tr, te, _, _ = load()
    df = sample_frame(tr, te, seed).reset_index(drop=True)
    n = len(df); t = (pd.to_datetime(df.timestamp) - T0).dt.total_seconds().values.astype(np.int64)
    codes = {c: pd.factorize(df[c])[0] for c in ['sender_id', 'receiver_id', 'device_id']}
    p2p = df.receiver_type.values == 'P2P'; A, B = [], []
    for k, g in [('device_id', gf.GAP['device_id']), ('sender_id', gf.GAP['sender_id'])]:
        a, b = gf.consecutive_links(codes[k], t, g); A.append(a); B.append(b)
    a, b = gf.consecutive_links(codes['receiver_id'], t, gf.GAP['receiver_p2p'], valid=p2p); A.append(a); B.append(b)
    nxt = gf.hop_next(df, t); ok = nxt >= 0; A.append(np.flatnonzero(ok)); B.append(nxt[ok])
    par = np.arange(n, dtype=np.int64); gf._union_all(par, np.concatenate(A).astype(np.int64), np.concatenate(B).astype(np.int64))
    return df.txn_id.values, par


def smooth(P, comp, alpha, fraud_only=True):
    """P' = (1-alpha) P + alpha * mean over component (excluding self) of P; singletons unchanged."""
    k = P.shape[1]
    S = np.zeros((comp.max() + 1, k)); np.add.at(S, comp, P)
    cnt = np.bincount(comp, minlength=comp.max() + 1)[comp]
    other = (S[comp] - P) / np.maximum(cnt - 1, 1)[:, None]
    Q = np.where((cnt > 1)[:, None], (1 - alpha) * P + alpha * other, P)
    if fraud_only:  # keep legit prob, only redistribute fraud mass by type
        fr = Q[:, 1:].sum(1, keepdims=True); fr0 = P[:, 1:].sum(1, keepdims=True)
        Q = np.concatenate([P[:, :1], Q[:, 1:] / np.maximum(fr, 1e-12) * fr0], axis=1)
    return Q
