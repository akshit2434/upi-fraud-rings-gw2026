import pandas as pd, numpy as np, sys
sys.path.insert(0,'/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/research/hunt')
from load import load, STRUCT, fauc
df=load()
df['k']=np.where(df.grp.isin(['legit','test']),df.grp,np.where(df.label.isin(STRUCT),'struct|'+df.grp,'single|'+df.grp))
df['hour']=df.ts.dt.hour; df['sec']=df.ts.dt.second; df['dow']=df.ts.dt.dayofweek; df['dom']=df.ts.dt.day
df['cents']=(df.amount*100).round()%100; df['own']=(df.device_id.str[2:]==df.sender_id.str[2:])
pd.set_option('display.width',250)
print(pd.crosstab(df.k, pd.cut(df.hour,[-1,5,8,11,14,17,20,23]),normalize='index').round(3))
print(pd.crosstab(df.k, df.dow,normalize='index').round(3))
print(df.groupby('k').agg(own=('own','mean'),cents0=('cents',lambda x:(x==0).mean()),p2p=('receiver_type',lambda x:(x=='P2P').mean()),n=('k','size')).round(3))
print(pd.crosstab(df.k, df.txn_type,normalize='index').round(3))
