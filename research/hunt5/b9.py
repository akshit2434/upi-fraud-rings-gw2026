import pandas as pd, numpy as np
df=pd.read_parquet('/tmp/hunt5/raw.parquet')
tiny=df[(df.receiver_type=='P2P')&(df.amount<=10)][['receiver_id','t','sender_id','label']].rename(columns={'receiver_id':'acct','t':'tt','sender_id':'lure_from','label':'lure_lab'})
x=df[(df.amount>10)&(df.is_test==0)][['txn_id','sender_id','receiver_id','t','label','conn','txn_type','receiver_type']]
x=x[x.label.isin(['fake_cashback','phishing_collect'])|((x.label=='legit')&(np.random.default_rng(0).random(len(x))<0.05))]
m=x.merge(tiny,left_on='sender_id',right_on='acct')
m=m[(m.t-m.tt>0)&(m.t-m.tt<7*86400)]
m['same_cp']=m.lure_from==m.receiver_id
m=m.sort_values('tt').groupby('txn_id').agg(any=('acct','size'),same=('same_cp','max'),dt=('tt',lambda s:0))
x['lure_any']=x.txn_id.isin(m.index); x['lure_same']=x.txn_id.map(m.same).fillna(False)
print(x.groupby(['label']).agg(n=('txn_id','size'),lure_any=('lure_any','mean'),lure_same=('lure_same','mean')).round(3))
print(x[x.label!='legit'].groupby(['label','conn']).agg(n=('txn_id','size'),lure_any=('lure_any','mean'),lure_same=('lure_same','mean')).round(3))
