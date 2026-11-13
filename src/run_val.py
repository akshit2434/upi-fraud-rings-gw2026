"""Validation: features on a 51%-density frame, train Jul-Aug, validate Sep."""
import sys, time, json
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import classification_report

sys.path.insert(0, 'src')
from features import load, build_features, sample_frame, LABELS
from common import macro_f1, tune_weights

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
drop = sys.argv[2].split(',') if len(sys.argv) > 2 and sys.argv[2] else []
t0 = time.time()
tr, te, acc, mer = load()
df = sample_frame(tr, te, seed)
X = build_features(df, acc, mer).drop(columns=['cat_mismatch'])
print('features', X.shape, round(time.time() - t0), 's')
X = X.drop(columns=[c for c in drop if c in X.columns])
lab = df.label.map({l: i for i, l in enumerate(LABELS)})
mon = df.timestamp.str[:7]
trn = (df._is_test == 0) & mon.isin(['2026-07', '2026-08'])
val = (df._is_test == 0) & (mon == '2026-09')

rng = np.random.default_rng(seed)
legit_keep = 0.25
mask = trn & ((lab != 0) | (rng.random(len(df)) < legit_keep))
w = np.where(lab[mask] == 0, 1 / legit_keep, 1.0)
params = dict(objective='multiclass', num_class=len(LABELS), learning_rate=0.05, num_leaves=63, min_child_samples=40,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=10, seed=seed)
dtr = lgb.Dataset(X[mask], lab[mask], weight=w)
dva = lgb.Dataset(X[val], lab[val], reference=dtr)
m = lgb.train(params, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
P = m.predict(X[val], num_iteration=m.best_iteration)
y = lab[val].values
print('best_iter', m.best_iteration, 'argmax macroF1', round(macro_f1(y, P.argmax(1)), 4))
wts, best = tune_weights(P, y)
print('tuned macroF1', round(best, 4), 'weights', np.round(wts, 2).tolist())
print(classification_report(y, (P * wts).argmax(1), target_names=LABELS, digits=3, zero_division=0))
imp = pd.Series(m.feature_importance('gain'), X.columns).sort_values(ascending=False)
print((imp / imp.sum()).round(4).head(40).to_string())
np.save(f'/tmp/val_P_{seed}.npy', P); np.save(f'/tmp/val_y_{seed}.npy', y)
print('total', round(time.time() - t0), 's')
