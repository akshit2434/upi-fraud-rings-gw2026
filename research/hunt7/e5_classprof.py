"""E5: distribution shift INSIDE the flagged set. For each predicted class, compare the profile of
Oct predicted rows (v12) with Sep predicted rows (v6 val) and with Sep TRUE rows (labels, 51% frame).
A predicted class whose Oct profile moved toward another class's profile is a typing-error suspect."""
import numpy as np, pandas as pd
from load import load, R, LABELS

d = load()
v12 = pd.read_csv(R + 'submissions/v12_tailfix.csv').rename(columns={'id': 'txn_id', 'label': 'v12'})
d = d.merge(v12, on='txn_id', how='left')
d['pred'] = np.where(d.grp == 'oct', d.v12, d.sp)
d['own'] = (d.device_is_own == 1).astype(int)
d['freshsim'] = (d.sim_age_hours < 72).astype(int)
d['newdev'] = (d.device_age_days < 2).astype(int)
d['night'] = d.hour.isin([0, 1, 2, 3, 4, 5]).astype(int)
d['eve'] = d.hour.isin([19, 20, 21, 22, 23]).astype(int)
d['p2p'] = (d.rt == 'P2P').astype(int)
for tt in ['PAY', 'QR', 'COLLECT', 'REFUND', 'AUTOPAY']: d['tt_' + tt] = (d.tt == tt).astype(int)
d['ecomgame'] = d.cat.isin(['ecommerce', 'gaming']).astype(int)
d['lending'] = (d.cat == 'lending').astype(int)
d['block'] = d.dnum.between(420000, 428454) | d.dnum.between(473632, 474551)
d['lamt'] = np.log10(d.amount)
P = ['p2p', 'tt_PAY', 'tt_QR', 'tt_COLLECT', 'tt_REFUND', 'ecomgame', 'lending', 'own', 'block', 'freshsim', 'newdev', 'night', 'eve',
     'away_from_home', 'pin_retries', 'amt_is_int', 'amt_round100', 'amt_just_under', 'amt_le10', 'lamt', 'amt_over_bal',
     'login_to_txn_sec', 'd_nacc_1h', 's_n_1h', 'r_nsend_1h', 'r_out_next1h', 'new_payee']
pd.set_option('display.width', 320); pd.set_option('display.max_columns', 40)
sep = d[d.grp == 'sep']; oc = d[d.grp == 'oct']
cnt = pd.DataFrame({'sep_true': sep.y.value_counts(), 'sep_pred': sep.pred.value_counts(), 'oct_pred': oc.pred.value_counts()})
cnt['oct/sep_pred'] = cnt.oct_pred / cnt.sep_pred
print(cnt.round(2).to_string())
rows = []
for c in LABELS[1:]:
    rows.append(sep[sep.y == c][P].mean().rename((c, 'sepTRUE')))
    rows.append(sep[sep.pred == c][P].mean().rename((c, 'sepPRED')))
    rows.append(oc[oc.pred == c][P].mean().rename((c, 'octPRED')))
T = pd.DataFrame(rows)
print(T.round(2).to_string())
T.to_csv('/tmp/hunt7/e5_prof.csv')
