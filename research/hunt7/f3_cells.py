"""F3: new/excess value combinations with binned numerics. 2- and 3-way cells over binned row fields;
Oct v12-unflagged count vs Sep v6-unflagged count (scaled). Aug (all rows, legit-labelled) as a noise control."""
import numpy as np, pandas as pd, itertools
from load import load, R
d = load()
v12 = pd.read_csv(R + 'submissions/v12_tailfix.csv').rename(columns={'id': 'txn_id', 'label': 'v12'})
d = d.merge(v12, on='txn_id', how='left')
d['flag'] = np.where(d.grp == 'oct', d.v12 != 'legit', d.flag)
B = pd.DataFrame(index=d.index)
B['tt'] = d.tt; B['rt'] = d.rt; B['cat'] = d.cat; B['qrm'] = d.qrm; B['app'] = d.app_s
B['dz'] = np.where(d.device_is_own == 1, 'own', np.where(d.dnum.between(420000, 428454) | d.dnum.between(473632, 474551), 'blk', 'oth'))
B['sim'] = pd.cut(d.sim_age_hours, [-1, 1, 24, 72, 720, 1e9], labels=['<1h', '<1d', '<3d', '<30d', 'old']).astype(str)
B['dev'] = pd.cut(d.device_age_days, [-1, 0.1, 1, 2, 30, 1e9], labels=['<.1', '<1', '<2', '<30', 'old']).astype(str)
B['pin'] = d.pin_retries.clip(0, 3).astype(int).astype(str)
B['login'] = pd.cut(d.login_to_txn_sec, [-1, 5, 15, 60, 300, 1e9], labels=['<5', '<15', '<60', '<300', 'long']).astype(str)
B['newp'] = d.new_payee.astype(int).astype(str); B['away'] = d.away_from_home.astype(int).astype(str)
B['hr'] = pd.cut(d.hour, [-1, 5, 11, 17, 23], labels=['night', 'morn', 'aft', 'eve']).astype(str)
B['amt'] = np.select([d.amt_le10 == 1, d.amt_just_under == 1, d.amt_round1000 == 1, d.amt_round100 == 1, d.amt_is_int == 1], ['le10', 'under', 'r1000', 'r100', 'int'], 'paise')
B['ab'] = pd.cut(d.amt_over_bal, [-1, 0.3, 0.9, 1.01, 1e9], labels=['<.3', '<.9', '~1', '>1']).astype(str)
B['kyc'] = d.sender_kyc.astype(str)
cols = list(B.columns)
C = {}; U = {}
for c in cols:
    C[c], U[c] = pd.factorize(B[c].astype(str))
so = (d.grp == 'sep') & (~d.flag.astype(bool)); oo = (d.grp == 'oct') & (~d.flag.astype(bool)); au = (d._month == '2026-08') & (d.y == 'legit')
scale = oo.sum() / so.sum(); scale_a = oo.sum() / au.sum()
fr = d.fraud.values
res = []
for k in (2, 3):
    for comb in itertools.combinations(cols, k):
        key = np.zeros(len(B), np.int64)
        for c in comb: key = key * 1000 + C[c]
        g = pd.DataFrame({'key': key, 's': so.values, 'o': oo.values, 'a': au.values, 'sf': (so & (fr == 1)).values}).groupby('key')[['s', 'o', 'a', 'sf']].sum()
        g = g[(g.o >= 12)]
        g['exp'] = g.s * scale; g['ratio'] = (g.o + 1) / (g.exp + 1); g['ratio_aug'] = (g.o + 1) / (g.a * scale_a + 1)
        g = g[(g.ratio >= 2.5) & (g.ratio_aug >= 2.5)]
        for kk, r in g.iterrows():
            parts = []; x = kk
            for c in reversed(comb): parts.append(U[c][x % 1000]); x //= 1000
            res.append({'fields': '+'.join(comb), 'cell': '|'.join(reversed(parts)), 'sep': int(r.s), 'oct': int(r.o), 'aug': int(r.a), 'ratio': round(r.ratio, 1),
                        'sepfraud': int(r.sf)})
r = pd.DataFrame(res).sort_values(['oct'], ascending=False)
r.to_csv('/tmp/hunt7/f3_cells.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300); pd.set_option('display.max_colwidth', 60)
print(len(r)); print(r.head(120).to_string())
