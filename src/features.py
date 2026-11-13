"""Feature engineering for UPI Fraud Rings (v2: edge-safe windows).

build_features(df, acc, mer) computes features for every row of df using ONLY the rows of df as
context. To mimic the test set (a ~51% random sample of October), callers pass a frame where the
training months are subsampled at the same rate (sample_frame).

Design rules (from adversarial validation): no unbounded 'all-time' aggregates and no unbounded
'time to next event' -- test rows sit at the end of the timeline, so such features drift.
Long-horizon windows look only into the past; short windows (<= 1 day) may look both ways.
"""
import numpy as np
import pandas as pd
from numba import njit

DATA = 'data/upi-fraud-rings'
T0 = pd.Timestamp('2026-07-01')
SHIFT = 10**8
LABELS = ['legit', 'mule_account', 'velocity_bot', 'layering_chain', 'sim_swap', 'account_takeover',
          'phishing_collect', 'qr_swap', 'refund_abuse', 'fake_cashback', 'loan_app_extortion',
          'merchant_collusion']
SAMPLE_RATE = 0.51
# Bump whenever build_features/neighbour_feats semantics change; caches are keyed by it.
# v3: features used for submissions up to v6_R1_bag (LB 0.4685). v4: r_nround_senders_7d sentinel fix.
FEATURE_VERSION = 'v4'
META = ['_label', '_month', '_is_test', '_txn_id']
DEFAULT_DROP = ['cat_mismatch', 'ip_rank_month']  # cat_mismatch is constant; ip_rank_month loses its signal in Oct (exp #15)
H, D = 3600, 86400
CAP = D  # cap for time-to-event features


def load():
    tr = pd.read_csv(f'{DATA}/train.csv')
    te = pd.read_csv(f'{DATA}/test.csv')
    acc = pd.read_csv(f'{DATA}/accounts.csv')
    mer = pd.read_csv(f'{DATA}/merchants.csv')
    return tr, te, acc, mer


def sample_frame(tr, te, seed, rate=SAMPLE_RATE):
    rng = np.random.default_rng(seed)
    keep = rng.random(len(tr)) < rate
    return pd.concat([tr[keep].assign(_is_test=0), te.assign(_is_test=1, label=np.nan)], ignore_index=True)


class Index:
    """Sorted (entity, time) index for window counts / sums / prev-next lookups."""

    def __init__(self, codes, t, values=None):
        key = codes.astype(np.int64) * SHIFT + t
        order = np.argsort(key, kind='stable')
        self.key = key[order]
        self.order = order
        self.cs = {}
        self.v = {}
        for name, v in (values or {}).items():
            v = np.asarray(v, dtype=np.float64)[order]
            self.v[name] = v
            self.cs[name] = np.concatenate([[0.0], np.cumsum(v)])

    def span(self, q, qt, lo, hi):
        base = q.astype(np.int64) * SHIFT + qt
        a = np.searchsorted(self.key, base + lo, 'left')
        b = np.searchsorted(self.key, base + hi, 'right')
        return a, b

    def count(self, q, qt, lo, hi):
        a, b = self.span(q, qt, lo, hi)
        return (b - a).astype(np.float64)

    def sum(self, name, a, b):
        return self.cs[name][b] - self.cs[name][a]

    def nxt(self, q, qt, name=None, strict=True, cap=CAP):
        base = q.astype(np.int64) * SHIFT + qt
        idx = np.searchsorted(self.key, base, 'right' if strict else 'left')
        idx2 = np.minimum(idx, len(self.key) - 1)
        dt = self.key[idx2] - base
        ok = (idx < len(self.key)) & (dt >= 0) & (dt <= cap)
        dt = np.where(ok, dt, np.nan)
        val = np.where(ok, self.v[name][idx2], np.nan) if name else None
        return dt, val

    def prv(self, q, qt, name=None, strict=True, cap=CAP):
        base = q.astype(np.int64) * SHIFT + qt
        idx = np.searchsorted(self.key, base, 'left' if strict else 'right') - 1
        idx2 = np.maximum(idx, 0)
        dt = base - self.key[idx2]
        ok = (idx >= 0) & (dt >= 0) & (dt <= cap)
        dt = np.where(ok, dt, np.nan)
        val = np.where(ok, self.v[name][idx2], np.nan) if name else None
        return dt, val


@njit(cache=True)
def _distinct_window(g, t, other, lo, hi):
    """Rows sorted by (g, t). For each row, # distinct `other` among same-g rows with t in [t+lo, t+hi]."""
    n = len(g)
    out = np.zeros(n, dtype=np.float64)
    start = 0
    while start < n:
        end = start
        while end < n and g[end] == g[start]:
            end += 1
        for i in range(start, end):
            seen = set()
            j = i
            while j >= start and t[j] >= t[i] + lo:
                seen.add(other[j]); j -= 1
            j = i + 1
            while j < end and t[j] <= t[i] + hi:
                seen.add(other[j]); j += 1
            out[i] = len(seen)
        start = end
    return out


def distinct_window(g, t, other, lo, hi):
    order = np.lexsort((t, g))
    res = _distinct_window(g[order], t[order], other[order].astype(np.int64), lo, hi)
    out = np.empty_like(res)
    out[order] = res
    return out


def _just_under(amt):
    out = np.full(len(amt), np.nan)
    for lim in [500, 1000, 2000, 2500, 5000, 10000, 15000, 20000, 25000, 50000, 100000]:
        gap = lim - amt
        rel = gap / lim
        m = (gap > 0) & (rel < 0.05) & (np.isnan(out) | (rel < out))
        out[m] = rel[m]
    return out


def build_features(df, acc, mer):
    df = df.reset_index(drop=True)
    n = len(df)
    ts = pd.to_datetime(df.timestamp)
    t = (ts - T0).dt.total_seconds().astype(np.int64).values
    F = {}

    ent = pd.Index(pd.unique(np.concatenate([df.sender_id.values, df.receiver_id.values, acc.account_id.values])))
    s = ent.get_indexer(df.sender_id).astype(np.int64)
    r = ent.get_indexer(df.receiver_id).astype(np.int64)
    dev = pd.factorize(df.device_id)[0].astype(np.int64)
    pkey = df.sender_id + '>' + df.receiver_id
    pidx = pd.Index(pd.unique(pkey))
    pair = pidx.get_indexer(pkey).astype(np.int64)
    rev = pidx.get_indexer(df.receiver_id + '>' + df.sender_id).astype(np.int64)
    amt = df.amount.values.astype(np.float64)
    is_p2p = (df.receiver_type == 'P2P').values
    month = ts.dt.month.values

    # ---------------- single-row ----------------
    hour = ts.dt.hour.values
    F['hour'] = hour
    F['minute_of_day'] = hour * 60 + ts.dt.minute.values
    F['weekday'] = ts.dt.weekday.values
    F['amount'] = amt
    F['log_amount'] = np.log1p(amt)
    F['amt_rank_month'] = pd.Series(amt).groupby(month).rank(pct=True).values
    F['amt_cents'] = np.round(amt % 1, 2)
    F['amt_is_int'] = (np.abs(amt - np.round(amt)) < 1e-6).astype(float)
    F['amt_round100'] = (np.abs(amt % 100) < 1e-6).astype(float)
    F['amt_round500'] = (np.abs(amt % 500) < 1e-6).astype(float)
    F['amt_round1000'] = (np.abs(amt % 1000) < 1e-6).astype(float)
    F['amt_just_under'] = _just_under(amt)
    F['amt_ends_99'] = ((np.round(amt) % 100) == 99).astype(float)
    F['amt_le10'] = (amt <= 10).astype(float)
    for c in ['sim_age_hours', 'device_age_days', 'sender_account_age_days', 'new_payee', 'pin_retries',
              'login_to_txn_sec', 'balance_before', 'qr_vpa_mismatch']:
        F[c] = df[c].values.astype(np.float64)
    F['qr_checked'] = df.qr_vpa_mismatch.notna().values.astype(float)
    bal = df.balance_before.values
    F['amt_over_bal'] = amt / (bal + 1)
    F['bal_after'] = bal - amt
    for c, cats in [('receiver_type', ['P2P', 'MERCHANT']), ('txn_type', ['PAY', 'QR', 'COLLECT', 'AUTOPAY', 'REFUND']),
                    ('app', ['APP_A', 'APP_B', 'APP_C', 'APP_D', 'APP_E'])]:
        F[c] = pd.Categorical(df[c], categories=cats).codes.astype(float)
    mcats = sorted(mer.category.unique())
    F['merchant_category'] = pd.Categorical(df.merchant_category, categories=mcats).codes.astype(float)
    F['ip_rank_month'] = df.groupby(month).ip_risk_score.rank(pct=True).values

    a = acc.set_index('account_id')
    a_open = pd.to_datetime(a.opened_on)
    home_s = df.sender_id.map(a.home_city)
    F['away_from_home'] = (df.device_city != home_s).values.astype(float)
    kyc = sorted(acc.kyc_level.unique())
    F['sender_kyc'] = pd.Categorical(df.sender_id.map(a.kyc_level), categories=kyc).codes.astype(float)
    F['sender_os'] = pd.Categorical(df.sender_id.map(a.os), categories=['android', 'ios']).codes.astype(float)
    F['recv_kyc'] = pd.Categorical(df.receiver_id.map(a.kyc_level), categories=kyc).codes.astype(float)
    F['recv_acc_age_days'] = (ts - df.receiver_id.map(a_open)).dt.days.values.astype(float)
    home_r = df.receiver_id.map(a.home_city)
    F['recv_same_city'] = np.where(is_p2p, (home_r == home_s).values.astype(float), np.nan)
    F['recv_city_eq_device'] = np.where(is_p2p, (home_r == df.device_city).values.astype(float), np.nan)
    m = mer.set_index('merchant_id')
    F['merchant_age_days'] = (ts - pd.to_datetime(df.receiver_id.map(m.onboarded_on))).dt.days.values.astype(float)
    mcity = df.receiver_id.map(m.city)
    F['merchant_city_eq_device'] = np.where(~is_p2p, (mcity == df.device_city).values.astype(float), np.nan)
    F['merchant_city_eq_home'] = np.where(~is_p2p, (mcity == home_s).values.astype(float), np.nan)
    F['cat_mismatch'] = np.where(~is_p2p, (df.receiver_id.map(m.category) != df.merchant_category).values.astype(float), np.nan)
    devown = df.device_id.str[2:].values == df.sender_id.str[2:].str.zfill(7).values
    F['device_is_own'] = devown.astype(float)  # DV0xxxxxx naming matches account number for the 'home' phone

    # ---------------- context ----------------
    is_ref = (df.txn_type == 'REFUND').values.astype(float)
    rnd = F['amt_round100']
    small = F['amt_le10']
    npy = F['new_payee']
    one = np.ones(n)
    night = ((hour >= 20) | (hour <= 5)).astype(float)
    by_s = Index(s, t, {'amt': amt, 'ref': is_ref, 'night': night, 'one': one, 'newp': npy})
    by_r = Index(r, t, {'amt': amt, 'newp': npy, 'round': rnd, 'one': one})
    by_d = Index(dev, t, {'amt': amt})
    by_p = Index(pair, t, {'amt': amt, 'small': small})

    def win(idx, q, name, lo, hi, self_in=True, sums=()):
        a_, b_ = idx.span(q, t, lo, hi)
        c = (b_ - a_).astype(np.float64) - (1 if self_in else 0)
        F[name] = c
        for sname, vself in sums:
            F[f'{name}_{sname}'] = idx.sum(sname, a_, b_) - (vself if self_in else 0)

    # sender as payer
    win(by_s, s, 's_n_10m', -600, 600)
    win(by_s, s, 's_n_1h', -H, H, sums=[('amt', amt)])
    win(by_s, s, 's_n_prev1h', -H, 0)
    win(by_s, s, 's_n_next1h', 0, H)
    win(by_s, s, 's_n_1d', -D, D, sums=[('amt', amt), ('newp', npy)])
    win(by_s, s, 's_n_p7d', -7 * D, 0, sums=[('ref', is_ref), ('night', night), ('amt', amt)])
    win(by_s, s, 's_n_p30d', -30 * D, 0, sums=[('ref', is_ref), ('amt', amt)])
    F['amt_vs_s_p30d'] = amt / ((F['s_n_p30d_amt'] + amt) / (F['s_n_p30d'] + 1))
    F['s_amt_1h_over_bal'] = (F['s_n_1h_amt'] + amt) / (bal + 1)
    F['s_dt_prev'], _ = by_s.prv(s, t)
    F['s_dt_next'], _ = by_s.nxt(s, t)

    # sender as receiver (mule onward transfers, layering hops)
    win(by_r, s, 's_in_prev1h', -H, 0, self_in=False, sums=[('amt', 0), ('newp', 0)])
    win(by_r, s, 's_in_1d', -D, D, self_in=False, sums=[('newp', 0)])
    win(by_r, s, 's_in_p30d', -30 * D, 0, self_in=False)
    F['s_last_in_dt'], v = by_r.prv(s, t, 'amt', strict=False)
    F['s_last_in_ratio'] = amt / v
    F['s_in_prev1h_ratio'] = amt / (F['s_in_prev1h_amt'] + 1)

    # receiver incoming
    win(by_r, r, 'r_n_10m', -600, 600)
    win(by_r, r, 'r_n_1h', -H, H, sums=[('newp', npy), ('amt', amt)])
    win(by_r, r, 'r_n_prev1h', -H, 0)
    win(by_r, r, 'r_n_1d', -D, D, sums=[('newp', npy), ('round', rnd)])
    win(by_r, r, 'r_n_p7d', -7 * D, 0, sums=[('round', rnd), ('newp', npy)])
    win(by_r, r, 'r_n_p30d', -30 * D, 0, sums=[('round', rnd), ('newp', npy)])
    F['r_round_share_p30d'] = F['r_n_p30d_round'] / (F['r_n_p30d'] + 1)
    F['r_newp_share_p30d'] = F['r_n_p30d_newp'] / (F['r_n_p30d'] + 1)
    F['r_1d_vs_p30d'] = (F['r_n_1d'] + 1) / (F['r_n_p30d'] / 30 + 1)
    F['amt_over_r_1h'] = amt / (F['r_n_1h_amt'] + amt)

    # receiver outgoing (forwarding)
    F['r_next_out_dt'], v = by_s.nxt(r, t, 'amt', strict=False)
    F['r_next_out_ratio'] = v / amt
    a_, b_ = by_s.span(r, t, 0, H)
    F['r_out_next1h'] = (b_ - a_).astype(float)
    F['r_out_next1h_ratio'] = by_s.sum('amt', a_, b_) / amt
    a_, b_ = by_s.span(r, t, 0, D)
    F['r_out_next1d'] = (b_ - a_).astype(float)
    win(by_s, r, 'r_out_p30d', -30 * D, 0, self_in=False)
    F['r_prev_out_dt'], _ = by_s.prv(r, t)

    # device
    win(by_d, dev, 'd_n_1m', -60, 60)
    win(by_d, dev, 'd_n_5m', -300, 300)
    win(by_d, dev, 'd_n_1h', -H, H)
    win(by_d, dev, 'd_n_1d', -D, D)
    F['d_dt_prev'], _ = by_d.prv(dev, t)
    F['d_dt_next'], _ = by_d.nxt(dev, t)
    F['d_nacc_1h'] = distinct_window(dev, t, s, -H, H)
    F['d_nacc_p7d'] = distinct_window(dev, t, s, -7 * D, 0)
    F['s_ndev_p7d'] = distinct_window(s, t, dev, -7 * D, 0)
    F['r_nsend_1h'] = distinct_window(r, t, s, -H, H)
    F['r_nsend_1d'] = distinct_window(r, t, s, -D, D)
    F['r_ndevcity_1d'] = distinct_window(r, t, pd.factorize(df.device_city)[0], -D, D)
    F['s_nrecv_1d'] = distinct_window(s, t, r, -D, D)

    # pair
    win(by_p, pair, 'p_n_1d', -D, D)
    win(by_p, pair, 'p_n_p7d', -7 * D, 0)
    win(by_p, pair, 'p_n_p30d', -30 * D, 0)
    F['p_dt_prev'], _ = by_p.prv(pair, t, cap=30 * D)
    F['p_share_r_p30d'] = (F['p_n_p30d'] + 1) / (F['r_n_p30d'] + 1)
    has_rev = rev >= 0
    rv = np.where(has_rev, rev, 0)
    dt, v = by_p.prv(rv, t, 'amt', cap=30 * D)
    F['rev_dt'] = np.where(has_rev, dt, np.nan)
    F['rev_amt'] = np.where(has_rev, v, np.nan)
    F['rev_ratio'] = amt / F['rev_amt']
    a_, b_ = by_p.span(rv, t, -30 * D, 0)
    F['rev_n_p30d'] = np.where(has_rev, b_ - a_, 0).astype(float)
    F['rev_small_p30d'] = np.where(has_rev, by_p.sum('small', a_, b_), 0)
    # tiny credit into sender from anyone
    sm = small > 0
    by_tiny = Index(r[sm], t[sm])
    F['s_tiny_in_dt'], _ = by_tiny.prv(s, t, strict=False, cap=7 * D)

    # receiver typical amount (past 30d, causal)
    win(by_r, r, 'r_p30d_amt', -30 * D, 0, self_in=True, sums=[('amt', amt)])
    F['amt_vs_r_p30d'] = amt / ((F['r_p30d_amt_amt'] + amt) / (F['r_p30d_amt'] + 1))
    del F['r_p30d_amt'], F['r_p30d_amt_amt']

    # ---- multi-day both-sided windows (refund / loan / collusion / cashback incidents span days) ----
    lend = (df.merchant_category == 'lending').values.astype(float)
    late = ((hour >= 19) | (hour <= 1)).astype(float)
    by_s2 = Index(s, t, {'ref': is_ref, 'lend': lend, 'late': late, 'round': rnd, 'one': one})
    for nm, lo, hi in [('7d', -7 * D, 7 * D), ('n7d', 0, 7 * D), ('3d', -3 * D, 3 * D)]:
        a_, b_ = by_s2.span(s, t, lo, hi)
        F[f's2_{nm}_n'] = (b_ - a_) - 1.0
        F[f's2_{nm}_ref'] = by_s2.sum('ref', a_, b_) - is_ref
        F[f's2_{nm}_lend'] = by_s2.sum('lend', a_, b_) - lend
        F[f's2_{nm}_late'] = by_s2.sum('late', a_, b_) - late
        F[f's2_{nm}_round'] = by_s2.sum('round', a_, b_) - rnd
    by_p2 = Index(pair, t, {'late': late, 'round': rnd, 'ref': is_ref})
    for nm, lo, hi in [('7d', -7 * D, 7 * D), ('n7d', 0, 7 * D), ('30d', -30 * D, 7 * D)]:
        a_, b_ = by_p2.span(pair, t, lo, hi)
        F[f'p2_{nm}_n'] = (b_ - a_) - 1.0
        F[f'p2_{nm}_late'] = by_p2.sum('late', a_, b_) - late
        F[f'p2_{nm}_round'] = by_p2.sum('round', a_, b_) - rnd
        F[f'p2_{nm}_ref'] = by_p2.sum('ref', a_, b_) - is_ref
    by_r2 = Index(r, t, {'round': rnd, 'ref': is_ref})
    a_, b_ = by_r2.span(r, t, -7 * D, 7 * D)
    F['r2_7d_round'] = by_r2.sum('round', a_, b_) - rnd
    F['r2_7d_ref'] = by_r2.sum('ref', a_, b_) - is_ref
    F['r2_7d_n'] = (b_ - a_) - 1.0
    F['r2_7d_round_share'] = F['r2_7d_round'] / (F['r2_7d_n'] + 1)
    F['p2_7d_round_share_of_r'] = F['p2_7d_round'] / (F['r2_7d_round'] + 1)
    # distinct round-amount senders to r within +-7d. Non-round rows map to sentinel -1, which distinct_window
    # counts as one extra value whenever any non-round row is in the window -> subtract it (v4 fix).
    raw = distinct_window(r, t, np.where(rnd > 0, s, -1), -7 * D, 7 * D)
    nonround_in_win = (F['r2_7d_n'] + 1) - (F['r2_7d_round'] + rnd)  # window rows incl. self minus round rows incl. self
    F['r_nround_senders_7d'] = raw - (nonround_in_win > 0)
    # tiny credit from this receiver to this sender within +-7d (cashback lure), and from anyone in next 1d before
    dt, v = by_p.nxt(rv, t, 'amt', cap=7 * D)
    F['rev_next_dt'] = np.where(has_rev, dt, np.nan)
    F['rev_next_amt'] = np.where(has_rev, v, np.nan)
    return pd.DataFrame(F).astype(np.float32)


def neighbour_feats(df, P, prefix='nb'):
    """Aggregate stage-1 class probabilities P (n x K, rows aligned with df) over each row's neighbourhoods.
    Self is excluded. Returns sums per fraud class (k>=1) and counts per neighbourhood."""
    df = df.reset_index(drop=True)
    n, K = P.shape
    t = (pd.to_datetime(df.timestamp) - T0).dt.total_seconds().astype(np.int64).values
    ent = pd.Index(pd.unique(np.concatenate([df.sender_id.values, df.receiver_id.values])))
    s = ent.get_indexer(df.sender_id).astype(np.int64)
    r = ent.get_indexer(df.receiver_id).astype(np.int64)
    dev = pd.factorize(df.device_id)[0].astype(np.int64)
    pair = pd.factorize(df.sender_id + '>' + df.receiver_id)[0].astype(np.int64)
    vals = {f'c{k}': P[:, k] for k in range(1, K)}
    vals['fraud'] = 1 - P[:, 0]
    by_s, by_r, by_d, by_p = Index(s, t, vals), Index(r, t, vals), Index(dev, t, vals), Index(pair, t, vals)
    F = {}
    names = list(vals)
    own = {k: v for k, v in vals.items()}
    specs = [  # (name, index, query codes, lo, hi, self_included)
        ('d10m', by_d, dev, -600, 600, True), ('d1d', by_d, dev, -D, D, True),
        ('s1h', by_s, s, -H, H, True), ('s1d', by_s, s, -D, D, True), ('s7d', by_s, s, -7 * D, 7 * D, True),
        ('r1h', by_r, r, -H, H, True), ('r1d', by_r, r, -D, D, True),
        ('p7d', by_p, pair, -7 * D, 7 * D, True),
        ('sin1h', by_r, s, -H, 0, False),     # payments INTO the sender in the previous hour
        ('rout1h', by_s, r, 0, H, False),     # payments OUT of the receiver in the next hour
        ('rin_as_s1d', by_s, r, -D, D, False),  # receiver's own outgoing activity
    ]
    for nm, idx, q, lo, hi, self_in in specs:
        a_, b_ = idx.span(q, t, lo, hi)
        cnt = (b_ - a_).astype(np.float64) - (1 if self_in else 0)
        F[f'{prefix}_{nm}_n'] = cnt
        for v in names:
            sm = idx.sum(v, a_, b_) - (own[v] if self_in else 0)
            F[f'{prefix}_{nm}_{v}'] = sm
            if v == 'fraud':
                F[f'{prefix}_{nm}_fraudmean'] = np.where(cnt > 0, sm / np.maximum(cnt, 1), np.nan)
    return pd.DataFrame(F).astype(np.float32)


def cache_path(seed, ver=None):
    return f'/tmp/feat_{ver or FEATURE_VERSION}_s{seed}.parquet'


def load_cached(seed, ver=None, build_if_missing=True):
    """Feature matrix + meta for a sampling seed, from a version-keyed cache (built and saved if missing)."""
    import os
    path = cache_path(seed, ver)
    if not os.path.exists(path):
        if not build_if_missing or (ver and ver != FEATURE_VERSION):
            raise FileNotFoundError(path)
        tr, te, acc, mer = load()
        df = sample_frame(tr, te, seed)
        X = build_features(df, acc, mer)
        X['_label'] = df.label.values; X['_month'] = df.timestamp.str[:7].values
        X['_is_test'] = df._is_test.values; X['_txn_id'] = df.txn_id.values
        X.to_parquet(path)
    X = pd.read_parquet(path)
    return X.drop(columns=META), X[META].reset_index(drop=True)


def apply_drop(X, spec='', default=True):
    """Drop columns by comma-separated names; a trailing '*' is a prefix wildcard. Same semantics everywhere."""
    names = [c for c in (spec.split(',') if isinstance(spec, str) else list(spec)) if c] + (DEFAULT_DROP if default else [])
    cols = [c for c in X.columns if any(c == d or (d.endswith('*') and c.startswith(d[:-1])) for d in names)]
    return X.drop(columns=cols)
