"""Ordered segmentation of a fraud device-ID block (hunt5).

The generator allocates fraud phones in one contiguous ID block with class sub-blocks in a FIXED order
(train: DV0420000-0428454, Oct: DV0473632-0474551). Labels inside the train block are ~clean (type noise ~3%).
The model mistypes block rows whose classes share a profile (sim_swap vs account_takeover: both new phone).
We recover the zone of each device with a monotone (left-to-right) Viterbi over the class order, using
  emission(device | zone c) = sum_rows log C[c, pred_row] + log Pn[c][min(n_rows,4)]
C = P(model prediction | true class) and Pn = rows-per-device distribution, both estimated on train-block rows.
"""
import numpy as np, pandas as pd

ORDER = ['mule_account', 'velocity_bot', 'layering_chain', 'sim_swap', 'account_takeover', 'phishing_collect',
         'qr_swap', 'refund_abuse', 'fake_cashback', 'loan_app_extortion', 'merchant_collusion']
LABS = ['legit'] + ORDER


def fit_emissions(lab, pred, dev, alpha=1.0):
    """lab/pred/dev: arrays for train-block rows with fraud labels (lab = true class, pred = model label)."""
    C = pd.crosstab(pd.Categorical(lab, ORDER), pd.Categorical(pred, LABS), dropna=False).reindex(index=ORDER, columns=LABS).fillna(0).values + alpha
    C = C / C.sum(1, keepdims=True)
    d = pd.DataFrame({'lab': lab, 'dev': dev})
    dl = d.groupby('dev').lab.agg(lambda s: s.value_counts().index[0]); dn = d.groupby('dev').size().clip(upper=4)
    Pn = np.full((len(ORDER), 4), 1.0)
    for i, c in enumerate(ORDER):
        v = dn[dl == c].value_counts()
        for k, x in v.items(): Pn[i, k - 1] += x
    Pn = Pn / Pn.sum(1, keepdims=True)
    return np.log(C), np.log(Pn)


def segment(dev, pred, logC, logPn, switch_pen=4.0, use_n=True, row_ll=None, row_w=1.0):
    """dev: device number per row, pred: model label per row. Returns dict device -> zone class (monotone in device order)."""
    d = pd.DataFrame({'dev': dev, 'p': pd.Categorical(pred, LABS).codes}).reset_index(drop=True)
    RL = np.zeros((len(d), len(ORDER))) if row_ll is None else row_w * np.asarray(row_ll)
    devs = np.sort(d.dev.unique())
    E = np.zeros((len(devs), len(ORDER)))
    g = d.groupby('dev')
    for j, (k, x) in enumerate(g):
        assert k == devs[j]
        E[j] = logC[:, x.p.values].sum(1) + (logPn[:, min(len(x), 4) - 1] if use_n else 0) + RL[x.index.values].sum(0)
    K = len(ORDER); n = len(devs)
    V = np.full((n, K), -np.inf); B = np.zeros((n, K), int)
    V[0] = E[0] - switch_pen * np.arange(K)  # allow starting anywhere (penalised)
    for j in range(1, n):
        best = np.maximum.accumulate(V[j - 1])            # best over zones <= k
        arg = np.array([np.argmax(V[j - 1][:k + 1]) for k in range(K)])
        stay = V[j - 1]; jump = np.r_[-np.inf, best[:-1]] - switch_pen
        jarg = np.r_[0, arg[:-1]]
        take_stay = stay >= jump
        V[j] = np.where(take_stay, stay, jump) + E[j]
        B[j] = np.where(take_stay, np.arange(K), jarg)
    z = np.zeros(n, int); z[-1] = V[-1].argmax()
    for j in range(n - 1, 0, -1): z[j - 1] = B[j, z[j]]
    return dict(zip(devs, np.array(ORDER)[z]))


def profile_key(df):
    """raw-row profile used as an extra zone emission: txn_type x receiver_type x integer-amount."""
    return (df.txn_type + '_' + df.receiver_type + '_' + (df.amount % 1 == 0).astype(int).astype(str)).values


def fit_profile(keys, zones, alpha=1.0):
    t = pd.crosstab(pd.Series(zones, name='z'), pd.Series(keys, name='k')).reindex(index=ORDER).fillna(0) + alpha
    return np.log(t / t.sum(1).values[:, None])


def profile_ll(keys, logT):
    """rows x zones log-lik; unseen keys get 0."""
    out = np.zeros((len(keys), len(ORDER)))
    cols = {k: i for i, k in enumerate(logT.columns)}
    M = logT.values
    for r, k in enumerate(keys):
        if k in cols: out[r] = M[:, cols[k]]
    return out
