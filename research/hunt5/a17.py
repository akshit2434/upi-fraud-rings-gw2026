import pandas as pd, numpy as np
P='/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/'
df=pd.read_parquet('/tmp/hunt5/raw.parquet')
v10=pd.read_csv(P+'submissions/v10_v6_octrules.csv').rename(columns={'id':'txn_id','label':'pred'})
df=df.merge(v10,on='txn_id',how='left')
df['dn']=df.device_id.str[2:].astype(int)
df['hour']=df.ts.dt.hour
def devtab(x,col):
    g=x.groupby('dn').agg(n=('txn_id','size'),s=('sender_id','nunique'),lab=(col,lambda s:''.join(sorted(set(v[:3] for v in s)))),
       sim=('sim_age_hours','median'),dage=('device_age_days','median'),amt=('amount','median'),bal=('balance_before','median'),pin=('pin_retries','mean'),login=('login_to_txn_sec','median'),hour=('hour','median'),span_h=('t',lambda v:(v.max()-v.min())/3600),p2p=('receiver_type',lambda v:(v=='P2P').mean()))
    return g
tr=df[(df.is_test==0)&(df.dn>=424400)&(df.dn<425300)]
g=devtab(tr,'label'); pd.set_option('display.width',250)
print(g.iloc[::6].to_string())
gs=devtab(df[(df.is_test==0)&(df.dn>=424539)&(df.dn<425039)],'label'); ga=devtab(df[(df.is_test==0)&(df.dn>=425039)&(df.dn<426783)],'label')
print('sim block devs', gs.describe().round(2).to_string()); print('ATO block devs', ga.describe().round(2).to_string())
te=df[(df.is_test==1)&(df.dn>=473880)&(df.dn<474300)]
print(devtab(te,'pred').to_string())
