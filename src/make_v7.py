"""v7 candidate: equal blend of LGB bag (4 frames) + CatBoost bag (2) + AutoGluon, event-smoothed (alpha .5);
class weights tuned on the same blend of Sep val probs. Writes preds/v7_blend_test_P.npy and submissions/v7_blend.csv."""
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from features import LABELS
from common import tune_weights
from smooth import comp_ids, smooth
ids0 = pd.read_csv('preds/lgb_s0_test_ids.csv').iloc[:, 0].values
def T(tag):
    assert (pd.read_csv(f'preds/{tag}_test_ids.csv').iloc[:, 0].values == ids0).all(); return np.load(f'preds/{tag}_test_P.npy')
L = np.mean([T(f'lgb_s{s}') for s in range(4)], 0); C = np.mean([T(f'cat_s{s}') for s in range(2)], 0)
A = pd.DataFrame(np.load('/tmp/ag/P_full.npy'), index=np.load('/tmp/ag/P_full_ids.npy', allow_pickle=True)).reindex(ids0).values
fid, comp = comp_ids(0); c = comp[pd.Series(np.arange(len(fid)), index=fid).reindex(ids0).values]
P = smooth((L + C + A) / 3, c, 0.5)
y = np.load('/tmp/val_y_s0.npy'); cv = np.load('/tmp/val_comp_s0.npy')
Lv = np.mean([np.load(f'/tmp/V_lgb_s{s}.npy') for s in range(4)], 0); Cv = np.mean([np.load(f'/tmp/V_cat_s{s}.npy') for s in range(2)], 0)
Pv = smooth((Lv + Cv + np.load('/tmp/ag/P_val.npy')) / 3, cv, 0.5)
w, f = tune_weights(Pv, y, iters=3); print('val same-set', round(f, 4), 'weights', np.round(w, 2).tolist())
np.save('preds/v7_blend_test_P.npy', P); pd.Series(ids0).to_csv('preds/v7_blend_test_ids.csv', index=False); np.save('preds/v7_blend_weights.npy', w)
te = pd.read_csv('data/upi-fraud-rings/test.csv', usecols=['txn_id'])
pred = pd.Series(np.array(LABELS)[(P * w).argmax(1)], index=ids0).reindex(te.txn_id).values
sub = pd.DataFrame({'id': te.txn_id, 'label': pred}); assert sub.label.notna().all()
sub.to_csv('submissions/v7_blend.csv', index=False)
v6 = pd.read_csv('submissions/v6_R1_bag.csv')
print('fraud share', round((sub.label != 'legit').mean(), 4), '| agreement with v6', round((v6.label.values == sub.label.values).mean(), 4),
      '| fraud rows changed vs v6', int(((v6.label != 'legit') | (sub.label != 'legit')).sum() - ((v6.label == sub.label) & (sub.label != 'legit')).sum()))
