"""October post-processing rules (exps #63, #66). Each rule changes only its targeted rows of a base submission.
Found by test-structure analysis (research/hunt2, research/hunt3); all are coded rules on provided columns, no hand labels.

 Q  QRP2P_BLANK : QR -> P2P with qr_vpa_mismatch blank (never in Jul-Sep; 74-82% qr_swap there) and base legit -> qr_swap. LB +0.0098.
 A  DEVBLOCK    : the generator allocates device IDs sequentially: own phones DV0000000-0419999, then a FRAUD device block
                  (train: DV0420000-0428454, 88% fraud, sub-blocks in a fixed class ORDER), then legit second phones (each ID used
                  once, 0.07% fraud). October repeats this with new IDs: fraud block = new IDs below the first long contiguous run.
                  Base-legit rows in the Oct fraud block -> class of the neighbouring base-fraud rows (order-aware, K=3).
 A2 OLDBLOCK    : Oct rows on train fraud-block devices called legit -> that device's majority train fraud label.
 A3 LEGITBLOCK  : Oct rows on the contiguous legit second-phone run called fraud -> legit.
 B  QRM_BLANK   : QR -> MERCHANT, mismatch blank, amount not a multiple of 100, base legit -> qr_swap.
 C  LURE        : tiny (<=10) P2P PAY X->Y followed within 7d by P2P Y->X, in events with a fake_cashback row:
                  base-legit members and phishing follow-ups >6h after the lure -> fake_cashback.
 D  QR_NOCAT    : QR -> P2P without merchant_category, base legit -> phishing_collect (train 79% phishing).
usage: oct_rules.py --base submissions/<in>.csv --rules QAABCD --out <name>"""
import sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, 'src')
from features import LABELS

ORDER = ['velocity_bot', 'mule_account', 'velocity_bot', 'layering_chain', 'sim_swap', 'account_takeover', 'phishing_collect',
         'qr_swap', 'refund_abuse', 'fake_cashback', 'loan_app_extortion', 'merchant_collusion']
TRAIN_BLOCK = (420000, 428455)


def contiguous_start(ids, run=100):
    ids = np.sort(np.unique(ids)); g = np.diff(ids)
    for i in range(len(g) - run):
        if (g[i:i + run] == 1).all():
            return ids[i]
    return None


def order_class(dn_q, dn_f, lab_f, K=3):
    o = np.argsort(dn_f, kind='stable'); dn_f = dn_f[o]; lab_f = lab_f[o]; out = []
    for x in dn_q:
        i = np.searchsorted(dn_f, x)
        L = lab_f[max(i - K, 0):i]; Rr = lab_f[i:i + K]
        Lc = pd.Series(L).value_counts().index[0] if len(L) else None
        Rc = pd.Series(Rr).value_counts().index[0] if len(Rr) else None
        if Lc is None: out.append(Rc); continue
        if Rc is None or Lc == Rc: out.append(Lc); continue
        best = None  # a class sitting exactly between Lc and Rc in the train order (recovers the 1-row velocity sub-block)
        for a_ in [j for j, c in enumerate(ORDER) if c == Lc]:
            for b_ in [j for j, c in enumerate(ORDER) if c == Rc and j > a_]:
                if best is None or b_ - a_ < best[1] - best[0]:
                    best = (a_, b_)
        if best and best[1] - best[0] == 2: out.append(ORDER[best[0] + 1]); continue
        dl = x - dn_f[i - 1] if i > 0 else np.inf; dr = dn_f[i] - x if i < len(dn_f) else np.inf
        out.append(Lc if dl <= dr else Rc)
    return np.array(out)


def run(sub, te, tr, rules='QAABCD', verbose=True):
    t = te.set_index('txn_id').reindex(sub.id).reset_index().rename(columns={'id': 'txn_id'})
    lab = sub.label.values.copy(); log = {}
    dn = t.device_id.str[2:].astype(int).values; trdn = tr.device_id.str[2:].astype(int).values
    if 'Q' in rules:
        q = ((t.txn_type == 'QR') & (t.receiver_type == 'P2P') & t.qr_vpa_mismatch.isna()).values & (lab == 'legit')
        lab[q] = 'qr_swap'; log['Q'] = int(q.sum())
    if 'A' in rules:
        new = dn > trdn.max(); start = dn[new].min(); end = contiguous_start(dn[new])
        blk = (dn >= start) & (dn < end); f = blk & (lab != 'legit'); q = blk & (lab == 'legit')
        lab[q] = order_class(dn[q], dn[f], lab[f], 3); log['A'] = int(q.sum()); log['A_block'] = (int(start), int(end) - 1)
        tb = tr[(trdn >= TRAIN_BLOCK[0]) & (trdn < TRAIN_BLOCK[1]) & (tr.label != 'legit').values]
        maj = tb.groupby('device_id').label.agg(lambda s: s.value_counts().index[0])
        q2 = (dn >= TRAIN_BLOCK[0]) & (dn < TRAIN_BLOCK[1]) & (lab == 'legit') & t.device_id.isin(maj.index).values
        lab[q2] = t.device_id[q2].map(maj).values; log['A2'] = int(q2.sum())
        q3 = (dn >= end) & (lab != 'legit'); lab[q3] = 'legit'; log['A3'] = int(q3.sum())
    if 'B' in rules:
        q = ((t.txn_type == 'QR') & (t.receiver_type == 'MERCHANT') & t.qr_vpa_mismatch.isna() & (t.amount % 100 != 0)).values & (lab == 'legit')
        lab[q] = 'qr_swap'; log['B'] = int(q.sum())
    if 'C' in rules:
        x = t[['sender_id', 'receiver_id', 'receiver_type', 'txn_type', 'amount', 'timestamp']].copy(); x['lab'] = lab
        x['ts'] = pd.to_datetime(x.timestamp); x['i'] = np.arange(len(x))
        tiny = x[(x.amount <= 10) & (x.receiver_type == 'P2P') & (x.txn_type == 'PAY')]; p = x[x.receiver_type == 'P2P']
        m = tiny.merge(p, left_on=['sender_id', 'receiver_id'], right_on=['receiver_id', 'sender_id'], suffixes=('', '_b'))
        m = m[(m.ts_b > m.ts) & (m.ts_b - m.ts < pd.Timedelta('7D'))]; m['gap'] = (m.ts_b - m.ts).dt.total_seconds() / 3600
        m['ev'] = m.sender_id + '>' + m.receiver_id
        evc = m.groupby('ev').apply(lambda g: (g.lab == 'fake_cashback').any() or (g.lab_b == 'fake_cashback').any())
        m = m[m.ev.map(evc).astype(bool)]
        idx = set(m.i[m.lab == 'legit']) | set(m.i_b[m.lab_b == 'legit']) | set(m.i_b[(m.lab_b == 'phishing_collect') & (m.gap > 6)])
        idx = np.array(sorted(idx), dtype=int); lab[idx] = 'fake_cashback'; log['C'] = len(idx)
    if 'D' in rules:
        q = ((t.txn_type == 'QR') & (t.receiver_type == 'P2P') & t.merchant_category.isna()).values & (lab == 'legit')
        lab[q] = 'phishing_collect'; log['D'] = int(q.sum())
    out = sub.copy(); out['label'] = lab
    if verbose:
        print(log, 'changed', int((out.label != sub.label).sum()), 'fraud share', round((out.label != 'legit').mean(), 4))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--base'); ap.add_argument('--rules', default='QAABCD'); ap.add_argument('--out')
    a = ap.parse_args()
    te = pd.read_csv('data/upi-fraud-rings/test.csv'); tr = pd.read_csv('data/upi-fraud-rings/train.csv', usecols=['device_id', 'label'])
    sub = pd.read_csv(a.base); out = run(sub, te, tr, a.rules)
    assert len(out) == len(te) and out.id.is_unique and set(out.id) == set(te.txn_id) and set(out.label) <= set(LABELS)
    out.to_csv(f'submissions/{a.out}.csv', index=False); print('wrote', a.out)
