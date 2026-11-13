"""Build relabel maps from OOF (seed-0 frame). Writes data/relabel_<name>.parquet (txn_id, new_label int).
R1: confident type swaps among {sim<->ato, mule<->layering, velocity->mule}. R2: R1 + legit rows with OOF max fraud prob > 0.8."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from features import cache_path, LABELS
L = {l: i for i, l in enumerate(LABELS)}
X = pd.read_parquet(cache_path(0), columns=['_label', '_is_test', '_txn_id'])
O = np.load('/tmp/oof_s0.npy'); tr = (X._is_test == 0).values
y = X._label.map(L).fillna(-1).astype(int).values
thr = np.array([np.nanmean(O[tr & (y == k), k]) for k in range(12)])
above = O >= thr
cand = np.where(above, O, -1); cl = cand.argmax(1); has = above.any(1) & tr
SW = {(L['sim_swap'], L['account_takeover']), (L['account_takeover'], L['sim_swap']), (L['mule_account'], L['layering_chain']),
      (L['layering_chain'], L['mule_account']), (L['velocity_bot'], L['mule_account'])}
r1 = np.array([has[i] and (y[i], cl[i]) in SW for i in range(len(y))])
fr = O[:, 1:].max(1); frc = O[:, 1:].argmax(1) + 1
r2u = tr & (y == 0) & (fr > 0.8)
for name, mask, newl in [('R1', r1, cl), ('R2', r1 | r2u, np.where(r1, cl, frc))]:
    out = pd.DataFrame({'txn_id': X._txn_id.values[mask], 'new_label': newl[mask].astype(int), 'old_label': y[mask]})
    out.to_parquet(f'data/relabel_{name}.parquet')
    print(name, len(out), out.groupby(['old_label', 'new_label']).size().rename(lambda i: i).to_dict() if len(out) < 0 else (out.old_label.map(dict(enumerate(LABELS))) + '->' + out.new_label.map(dict(enumerate(LABELS)))).value_counts().head(12).to_dict())
