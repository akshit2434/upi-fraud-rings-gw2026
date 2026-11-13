"""Run in .venv-ag: AutoGluon tabular on /tmp/ag/<mode>_train.parquet -> probs for <mode>_pred rows.
usage: .venv-ag/bin/python src/ag_run.py val|full TIME_LIMIT_SEC"""
import sys, numpy as np, pandas as pd
from autogluon.tabular import TabularPredictor
mode, tl = sys.argv[1], int(sys.argv[2])
T = pd.read_parquet(f'/tmp/ag/{mode}_train.parquet'); Pr = pd.read_parquet(f'/tmp/ag/{mode}_pred.parquet')
pred = TabularPredictor(label='label', sample_weight='w', eval_metric='log_loss', path=f'/tmp/ag/model_{mode}', problem_type='multiclass')
pred.fit(T, time_limit=tl, presets='medium_quality', num_cpus=10,
         hyperparameters={'GBM': [{}, {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}}], 'CAT': {}, 'XGB': {}, 'RF': {}, 'XT': {}})
print(pred.leaderboard(silent=True).to_string())
P = pred.predict_proba(Pr.drop(columns=['_txn_id']))
P = P[sorted(P.columns)].values
np.save(f'/tmp/ag/P_{mode}.npy', P); np.save(f'/tmp/ag/P_{mode}_ids.npy', Pr._txn_id.values)
print('saved', P.shape)
