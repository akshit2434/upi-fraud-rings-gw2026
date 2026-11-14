"""F1: near-miss band excess. Per class, count v12/v6-legit rows with moderate model probability for that class,
Oct (v6 test P) vs Sep (v6 bagged val P, model trained on Jul-Aug). An October pile-up below threshold = partial adaptation."""
import numpy as np, pandas as pd
from load import load, R, LABELS
d = load()
v12 = pd.read_csv(R + 'submissions/v12_tailfix.csv').rename(columns={'id': 'txn_id', 'label': 'v12'})
Pt = np.load(R + 'preds/v6_R1_bag_test_P.npy'); it = pd.read_csv(R + 'preds/v6_R1_bag_test_ids.csv').iloc[:, 0].values
Ps = np.load('/tmp/P_bag4_R1.npy'); is_ = np.load('/tmp/P_bag4_R1_ids.npy', allow_pickle=True)
w = np.load(R + 'preds/v6_R1_bag_weights.npy')
T = pd.DataFrame(Pt, columns=LABELS); T['txn_id'] = it; T = T.merge(v12, on='txn_id')
S = pd.DataFrame(Ps, columns=LABELS); S['txn_id'] = is_; S = S.merge(d[['txn_id', 'y']], on='txn_id')
S['pred'] = np.array(LABELS)[(Ps * w).argmax(1)]; T['pred'] = T.v12
bands = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
rows = []
for c in LABELS[1:]:
    for lo, hi in zip(bands[:-1], bands[1:]):
        so = S[(S.pred == 'legit') & (S[c] >= lo) & (S[c] < hi)]; to = T[(T.pred == 'legit') & (T[c] >= lo) & (T[c] < hi)]
        rows.append({'cls': c, 'band': f'{lo}-{hi}', 'sep': len(so), 'oct': len(to), 'oct-sep': len(to) - len(so),
                     'sep_fraud%': round(100 * (so.y != 'legit').mean(), 1), 'sep_cls%': round(100 * (so.y == c).mean(), 1)})
pd.set_option('display.width', 200); pd.set_option('display.max_rows', 200)
r = pd.DataFrame(rows); print(r.to_string())
pf_s = 1 - S[S.pred == 'legit'].legit; pf_t = 1 - T[T.pred == 'legit'].legit
for lo in [0.02, 0.05, 0.1, 0.2, 0.3]:
    print(f'P(fraud)>={lo}: sep {int((pf_s >= lo).sum())} oct {int((pf_t >= lo).sum())}  sep fraud% among them {100 * (S[(S.pred == "legit") & (1 - S.legit >= lo)].y != "legit").mean():.1f}')
