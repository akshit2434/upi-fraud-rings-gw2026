"""hunt7 base frame: v4 features (seed-0 51% frame) + raw columns + labels + conn + model flags.
Sep flag = v6 bagged val prediction (model trained on Jul-Aug); Oct flag = v11_blockseg label.
Cached in the session scratchpad (BASE)."""
import numpy as np, pandas as pd, os
R = '/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/'
S = os.environ.get('H7', '/tmp/hunt7')
os.makedirs(S, exist_ok=True)
BASE = S + '/base.parquet'
LABELS = ['legit', 'mule_account', 'velocity_bot', 'layering_chain', 'sim_swap', 'account_takeover', 'phishing_collect',
          'qr_swap', 'refund_abuse', 'fake_cashback', 'loan_app_extortion', 'merchant_collusion']
STRUCT = ['mule_account', 'velocity_bot', 'layering_chain', 'sim_swap', 'account_takeover', 'merchant_collusion']


def load():
    if os.path.exists(BASE):
        return pd.read_parquet(BASE)
    f = pd.read_parquet('/tmp/feat_v4_s0.parquet')
    tr = pd.read_csv(R + 'data/upi-fraud-rings/train.csv')
    te = pd.read_csv(R + 'data/upi-fraud-rings/test.csv')
    raw = pd.concat([tr, te.assign(label=np.nan)], ignore_index=True).drop(columns=['app_version'])
    keep = ['txn_id', 'timestamp', 'sender_id', 'receiver_id', 'device_id', 'device_city', 'label', 'ip_risk_score']
    rawc = raw[['txn_id', 'receiver_type', 'txn_type', 'merchant_category', 'qr_vpa_mismatch', 'app']].rename(
        columns={'receiver_type': 'rt', 'txn_type': 'tt', 'merchant_category': 'cat', 'qr_vpa_mismatch': 'qrm', 'app': 'app_s'})
    rawc['cat'] = rawc.cat.fillna('NA'); rawc['qrm'] = rawc.qrm.map({0.0: '0', 1.0: '1'}).fillna('NA')
    d = f.merge(raw[keep], left_on='_txn_id', right_on='txn_id', how='left').merge(rawc, on='txn_id', how='left')
    conn = pd.read_parquet(R + 'data/conn.parquet')
    d = d.merge(conn, on='txn_id', how='left')
    P = np.load('/tmp/P_bag4_R1.npy'); ids = np.load('/tmp/P_bag4_R1_ids.npy', allow_pickle=True)
    w = np.load(R + 'preds/v6_R1_bag_weights.npy')
    sp = pd.DataFrame({'txn_id': ids, 'sp': np.array(LABELS)[(P * w).argmax(1)], 'sp_pf': 1 - P[:, 0]})
    d = d.merge(sp, on='txn_id', how='left')
    v11 = pd.read_csv(R + 'submissions/v11_blockseg.csv').rename(columns={'id': 'txn_id', 'label': 'v11'})
    d = d.merge(v11, on='txn_id', how='left')
    d['ts'] = pd.to_datetime(d.timestamp)
    d['t'] = (d.ts - pd.Timestamp('2026-07-01')).dt.total_seconds()
    d['grp'] = np.where(d._is_test == 1, 'oct', np.where(d._month == '2026-09', 'sep', 'julaug'))
    d.loc[(d._is_test == 0) & d._month.isin(['2026-10']), 'grp'] = 'trOct'
    d['flag'] = np.where(d.grp == 'oct', d.v11 != 'legit', np.where(d.grp == 'sep', d.sp != 'legit', False))
    d['y'] = d.label.where(d._is_test == 0)
    d['fraud'] = (d.y.notna() & (d.y != 'legit')).astype(int)
    d['dnum'] = d.device_id.str[2:].astype(int)
    d = d.drop(columns=['timestamp'])
    d.to_parquet(BASE)
    return d


if __name__ == '__main__':
    d = load()
    print(d.shape, d.grp.value_counts().to_dict())
    s = d[d.grp == 'sep']
    print('sep flag rate', s.flag.mean(), 'fraud', s.fraud.mean(), 'unflagged fraud', s[~s.flag].fraud.mean())
    o = d[d.grp == 'oct']; print('oct flag rate', o.flag.mean())
