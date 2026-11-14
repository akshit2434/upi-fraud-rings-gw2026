"""Post-processing rules for October drift found in test structure (exp #63).
R_QRP2P_NULL: QR payment to a P2P account with qr_vpa_mismatch blank -> qr_swap.
  Train: QR->P2P never has a blank mismatch (0/2,199) and is 74-82% qr_swap for mismatch 0 or 1.
  Test: 140/319 QR->P2P rows are blank (rings dodging the check); their receivers have 39.5% fraud history in train.
usage: rules.py --sub submissions/<in>.csv --out <name>"""
import argparse, pandas as pd


def apply_rules(sub, te):
    t = te.set_index('txn_id').reindex(sub.id)
    m = ((t.txn_type == 'QR') & (t.receiver_type == 'P2P') & t.qr_vpa_mismatch.isna()).values
    out = sub.copy(); out.loc[m & (out.label == 'legit').values, 'label'] = 'qr_swap'
    return out, int(m.sum())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--sub'); ap.add_argument('--out'); a = ap.parse_args()
    te = pd.read_csv('data/upi-fraud-rings/test.csv')
    sub, n = apply_rules(pd.read_csv(a.sub), te)
    sub.to_csv(f'submissions/{a.out}.csv', index=False)
    print(a.out, 'rule rows', n, 'fraud share', round((sub.label != 'legit').mean(), 4))
