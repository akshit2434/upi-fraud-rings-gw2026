"""Multi-model validation: train Jul-Aug on sampling frame --tseed, predict the SEED-0 Sep rows (common rows for
blending/bagging). Saves /tmp/V_<tag>.npy (+ _ids.npy). With --full also trains on all training months of frame
--tseed and saves test probs preds/<tag>_test_P.npy (+ ids, manifest) with identical settings.
usage: exp2.py --model lgb|cat --tseed 0 --tag NAME [--full] [--extra graph]"""
import sys, time, json, argparse
import numpy as np, pandas as pd
sys.path.insert(0, 'src')
from features import LABELS, FEATURE_VERSION, load_cached, apply_drop
from common import macro_f1, tune_weights, xfit_f1, get_conn, clean_bad, apply_relabel

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='lgb'); ap.add_argument('--tseed', type=int, default=0)
ap.add_argument('--drop', default='ip_rank_month'); ap.add_argument('--clean', default='iso7')
ap.add_argument('--relabel', default='data/relabel_R1.parquet'); ap.add_argument('--legit_keep', type=float, default=0.25)
ap.add_argument('--iters', type=int, default=400); ap.add_argument('--lr', type=float, default=0.05)
ap.add_argument('--leaves', type=int, default=63); ap.add_argument('--depth', type=int, default=8)
ap.add_argument('--extra', default='', help='comma list of extra feature blocks: graph')
ap.add_argument('--full', action='store_true'); ap.add_argument('--novalid', action='store_true')
ap.add_argument('--threads', type=int, default=10); ap.add_argument('--tag', required=True)
a = ap.parse_args()
L2I = {l: i for i, l in enumerate(LABELS)}


def load(seed):
    X, meta = load_cached(seed)
    for e in [x for x in a.extra.split(',') if x]:
        G = pd.read_parquet(f'/tmp/{e}_{FEATURE_VERSION}_s{seed}.parquet')
        assert (G._txn_id.values == meta._txn_id.values).all()
        X = pd.concat([X, G.drop(columns=['_txn_id'])], axis=1)
    return apply_drop(X, a.drop), meta


def fit(X, y, w, seed):
    if a.model == 'lgb':
        import lightgbm as lgb
        p = dict(objective='multiclass', num_class=12, learning_rate=a.lr, num_leaves=a.leaves, min_child_samples=40,
                 feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=a.threads, seed=seed)
        m = lgb.train(p, lgb.Dataset(X, y, weight=w), a.iters)
        return m.predict
    if a.model == 'cat':
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(loss_function='MultiClass', iterations=a.iters, learning_rate=a.lr, depth=a.depth,
                               thread_count=a.threads, random_seed=seed, verbose=200, border_count=128)
        m.fit(X.values, y, sample_weight=w)
        return lambda Z: m.predict_proba(Z.values)
    raise ValueError(a.model)


def train_mask(lab, trn, bad, seed):
    rng = np.random.default_rng(seed)
    mask = trn & ~bad & ((lab != 0) | (rng.random(len(lab)) < a.legit_keep))
    return mask, np.where(lab[mask] == 0, 1 / a.legit_keep, 1.0)


t0 = time.time()
X, meta = load(a.tseed)
ids = meta._txn_id.values
lab0 = meta._label.map(L2I).fillna(-1).astype(int).values
res = {}
if not a.novalid:
    trn = ((meta._is_test == 0) & meta._month.isin(['2026-07', '2026-08'])).values
    bad = clean_bad(lab0, get_conn(ids, until='2026-09-01'), a.clean)
    lab = apply_relabel(lab0, ids, a.relabel, mask=trn)
    mask, w = train_mask(lab, trn, bad, a.tseed)
    pred = fit(X[mask], lab[mask], w, a.tseed)
    if a.tseed == 0:
        X0, m0 = X, meta
    else:
        X0, m0 = load(0)
    sep = ((m0._is_test == 0) & (m0._month == '2026-09')).values
    P = pred(X0[sep][X.columns]); y = m0._label[sep].map(L2I).values.astype(int)
    np.save(f'/tmp/V_{a.tag}.npy', P); np.save(f'/tmp/V_{a.tag}_ids.npy', m0._txn_id.values[sep])
    res = dict(argmax=round(macro_f1(y, P.argmax(1)), 4), sameset=round(tune_weights(P, y, iters=2)[1], 4), xfit=round(xfit_f1(P, y), 4))
    print(a.tag, 'VAL', res, f'{time.time()-t0:.0f}s', flush=True)
if a.full:
    trn = (meta._is_test == 0).values
    bad = clean_bad(lab0, get_conn(ids), a.clean)
    lab = apply_relabel(lab0, ids, a.relabel, mask=trn)
    mask, w = train_mask(lab, trn, bad, a.tseed)
    pred = fit(X[mask], lab[mask], w, a.tseed)
    tm = (meta._is_test == 1).values; tid = ids[tm]; o = np.argsort(tid)
    np.save(f'preds/{a.tag}_test_P.npy', pred(X[tm])[o]); pd.Series(tid[o]).to_csv(f'preds/{a.tag}_test_ids.csv', index=False)
    json.dump(dict(vars(a), feature_version=FEATURE_VERSION, val=res, columns=list(X.columns)), open(f'preds/{a.tag}_manifest.json', 'w'), indent=1)
    print(a.tag, 'FULL saved', f'{time.time()-t0:.0f}s', flush=True)
