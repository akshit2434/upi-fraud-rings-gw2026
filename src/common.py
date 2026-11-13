import numpy as np
from sklearn.metrics import f1_score

from features import LABELS


def macro_f1(y, pred):
    return f1_score(y, pred, average='macro', labels=list(range(len(LABELS))))


def tune_weights(P, y, iters=3, grid=None):
    """Coordinate ascent on per-class multipliers applied to probabilities (argmax decision)."""
    k = P.shape[1]
    w = np.ones(k)
    grid = grid if grid is not None else np.exp(np.linspace(np.log(0.05), np.log(20), 41))
    best = macro_f1(y, (P * w).argmax(1))
    for _ in range(iters):
        for c in range(1, k):
            bw = w[c]
            for g in grid:
                w[c] = g
                f = macro_f1(y, (P * w).argmax(1))
                if f > best:
                    best, bw = f, g
            w[c] = bw
    return w, best


# ---------------- shared policies (used by exp.py AND train_full.py so they can't diverge) ----------------
import pandas as pd

STRUCT_CLASSES = ['mule_account', 'velocity_bot', 'layering_chain', 'sim_swap', 'account_takeover', 'merchant_collusion', 'fake_cashback']
CLEAN_POLICIES = ('none', 'iso7', 'isoall')


def get_conn(ids, until=None):
    """Connectivity flags; until='2026-09-01' uses only labels before that date (for Jul-Aug -> Sep validation)."""
    path = f'data/conn_until_{until}.parquet' if until else 'data/conn.parquet'
    return pd.read_parquet(path).set_index('txn_id').conn.reindex(ids).fillna(False).values.astype(bool)


def clean_bad(lab, conn, policy):
    """Rows to drop from training. lab: int labels (-1 = unlabelled)."""
    from features import LABELS
    if policy not in CLEAN_POLICIES:
        raise ValueError(f'unknown clean policy {policy}')
    iso = (lab > 0) & ~conn
    if policy == 'iso7':
        return iso & np.isin(lab, [LABELS.index(c) for c in STRUCT_CLASSES])
    if policy == 'isoall':
        return iso
    return np.zeros(len(lab), bool)


def apply_relabel(lab, ids, path, mask=None):
    """Replace labels from a relabel map (txn_id -> new_label) on rows in mask (default: all)."""
    if not path:
        return lab
    rl = pd.read_parquet(path).set_index('txn_id').new_label.reindex(ids).values
    ok = ~np.isnan(rl) & (mask if mask is not None else True)
    out = lab.copy(); out[ok] = rl[ok].astype(int)
    return out


def xfit_f1(P, y, seed=123):
    """Cross-fitted macro-F1: tune class weights on one random half, score the other, and vice versa."""
    half = np.random.default_rng(seed).random(len(y)) < 0.5
    wa, _ = tune_weights(P[half], y[half], iters=2); wb, _ = tune_weights(P[~half], y[~half], iters=2)
    pr = np.empty(len(y), int); pr[~half] = (P[~half] * wa).argmax(1); pr[half] = (P[half] * wb).argmax(1)
    return macro_f1(y, pr)
