import pandas as pd, numpy as np
P='/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/'
df=pd.read_parquet('/tmp/hunt5/raw.parquet')
v10=pd.read_csv(P+'submissions/v10_v6_octrules.csv').rename(columns={'id':'txn_id','label':'pred'})
te=df[df.is_test==1].merge(v10,on='txn_id')
te['own']=te.device_id.str[2:]==te.sender_id.str[2:]
te['hour']=te.ts.dt.hour
V=te[te.pred=='velocity_bot'].copy()
V['dev_n']=V.groupby('device_id').txn_id.transform('size'); V['dev_s']=V.groupby('device_id').sender_id.transform('nunique')
print(len(V), 'own',V.own.mean(), 'cats',V.merchant_category.value_counts().head(8).to_dict())
print(V.groupby('device_id').agg(n=('txn_id','size'),s=('sender_id','nunique'),r=('receiver_id','nunique'),cat=('merchant_category',lambda x:x.value_counts().index[0] if x.notna().any() else 'NA'),
  amt=('amount','median'),login=('login_to_txn_sec','median'),hour=('hour','median'),simage=('sim_age_hours','median'),devage=('device_age_days','median'),own=('own','mean'),span_min=('t',lambda x:(x.max()-x.min())/60)).sort_values('n',ascending=False).head(40).to_string())
# train connected velocity profile
tr=df[(df.is_test==0)&(df.label=='velocity_bot')&df.conn]
tr=tr.assign(own=tr.device_id.str[2:]==tr.sender_id.str[2:])
g=tr.groupby('device_id').agg(n=('txn_id','size'),s=('sender_id','nunique'),login=('login_to_txn_sec','median'),amt=('amount','median'),own=('own','mean'),span=('t',lambda x:(x.max()-x.min())/60))
print('train conn velocity devices'); print(g.describe().round(1))
