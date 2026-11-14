import numpy as np
from sklearn.metrics import f1_score

from features import LABELS


def macro_f1(y, pred):
    """Macro-F1 over all 12 labels (same as sklearn f1_score(average='macro', labels=range(12)), numpy-fast)."""
    k = len(LABELS)
    cm = np.bincount(np.asarray(y) * k + np.asarray(pred), minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float); den = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)))


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
CLEAN_POLICIES = ('none', 'iso7', 'isoall', 'iso7oof', 'iso7oofu')


def get_conn(ids, until=None):
    """Connectivity flags; until='2026-09-01' uses only labels before that date (for Jul-Aug -> Sep validation)."""
    path = f'data/conn_until_{until}.parquet' if until else 'data/conn.parquet'
    return pd.read_parquet(path).set_index('txn_id').conn.reindex(ids).fillna(False).values.astype(bool)


def clean_bad(lab, conn, policy, ids=None):
    """Rows to drop from training. lab: int labels (-1 = unlabelled).
    iso7oof = iso7 + fraud rows whose week-blocked OOF P(legit) > 0.8 (data/oofbad80.parquet; legit-profile rows, exp #93)."""
    from features import LABELS
    if policy not in CLEAN_POLICIES:
        raise ValueError(f'unknown clean policy {policy}')
    iso = (lab > 0) & ~conn
    if policy == 'iso7':
        return iso & np.isin(lab, [LABELS.index(c) for c in STRUCT_CLASSES])
    if policy == 'isoall':
        return iso
    if policy in ('iso7oof', 'iso7oofu'):  # oofu also drops legit-labelled rows with OOF P(legit) < 0.1 (unreported fraud)
        ob = np.isin(ids, pd.read_parquet('data/oofbadfull_u.parquet' if policy == 'iso7oofu' else 'data/oofbad80.parquet').txn_id.values)
        if policy == 'iso7oofu':
            return (iso & np.isin(lab, [LABELS.index(c) for c in STRUCT_CLASSES])) | ((lab >= 0) & ob)
        return (iso & np.isin(lab, [LABELS.index(c) for c in STRUCT_CLASSES])) | ((lab > 0) & ob)
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
