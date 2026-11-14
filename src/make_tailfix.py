"""hunt5 v12: October fraud-block tail fixes on top of v11 (src/make_blockseg.py output).
Train tail is strictly ordered refund -> cashback -> loan -> collusion and collusion-looking rows inside the train refund
zone are labelled refund (8/8). So: (1) rows the base calls merchant_collusion inside the Oct refund zone -> refund_abuse;
(2) Oct devices >= first confident collusion call after the last confident loan call (DV0474529) -> merchant_collusion.
usage: .venv/bin/python src/make_tailfix.py --base submissions/v11_blockseg.csv --out submissions/v12_tailfix.csv [--parts 1,2]"""
import argparse, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument('--base', default='submissions/v11_blockseg.csv')
ap.add_argument('--out', default='submissions/v12_tailfix.csv'); ap.add_argument('--parts', default='1,2')
ap.add_argument('--refund_zone', default='474462,474497'); ap.add_argument('--coll_start', type=int, default=474529)
a = ap.parse_args(); parts = a.parts.split(',')
te = pd.read_csv('data/upi-fraud-rings/test.csv', usecols=['txn_id', 'device_id']); te['dn'] = te.device_id.str[2:].astype(int)
b = pd.read_csv(a.base).set_index('id').label; s = b.copy(); dn = te.set_index('txn_id').dn.reindex(s.index)
r0, r1 = map(int, a.refund_zone.split(','))
if '1' in parts:
    m = (dn >= r0) & (dn <= r1) & (s == 'merchant_collusion'); s[m] = 'refund_abuse'; print('part1 collusion->refund', int(m.sum()))
if '2' in parts:
    m = (dn >= a.coll_start) & (dn < 474552); print('part2 ->collusion from', s[m].value_counts().to_dict()); s[m] = 'merchant_collusion'
print('changed', int((s != b).sum()))
out = s.reset_index(); out.columns = ['id', 'label']; assert len(out) == len(te) and out.id.is_unique
out.to_csv(a.out, index=False); print('wrote', a.out)
