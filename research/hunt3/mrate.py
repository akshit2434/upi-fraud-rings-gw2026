from base import *
tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv'); c=pd.read_parquet(R+'data/conn.parquet'); tr=tr.merge(c,on='txn_id')
tr=tr[tr.txn_type!='QR']; tr['mk']=tr.qr_vpa_mismatch.notna(); tr['fraud']=tr.label!='legit'
tr['hr']=tr.timestamp.str[11:13].astype(int)
f=tr[tr.fraud]
print('by class x conn'); print(f.groupby(['label','conn']).mk.mean().unstack().round(3))
print('by type'); print(f.groupby('txn_type').mk.agg(['mean','size']).round(3))
print('by recv'); print(f.groupby('receiver_type').mk.agg(['mean','size']).round(3))
print('by hour band'); print(f.groupby(pd.cut(f.hr,[-1,5,11,17,23])).mk.mean().round(3))
print('by amount q'); print(f.groupby(pd.qcut(f.amount,5)).mk.mean().round(3))
print('by app'); print(f.groupby('app').mk.mean().round(3)); print(f.groupby('app_version').mk.mean().round(3))
print('legit mk rate', tr[~tr.fraud].mk.mean(), 'n', tr[~tr.fraud].mk.sum())
print('legit mk by type'); print(tr[~tr.fraud].groupby('txn_type').mk.mean())
# legit marked rows: look like?
L=tr[(~tr.fraud)&tr.mk]; print(L.groupby('app_version').size()); print(L[['amount','sim_age_hours','device_age_days','login_to_txn_sec','new_payee']].describe().round(1))
