import pandas as pd, numpy as np, sys
sys.path.insert(0,'/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/research/hunt')
from load import load, STRUCT
df=load(); df['dn']=df.device_id.str[2:].astype(int); df['sn']=df.sender_id.str[2:].astype(int)
x=df[(df.dn!=df.sn)&(df.dn<420000)].copy()
x['k']=np.where(x.is_test==1,'test',np.where(x.label=='legit','legit',np.where(x.label.isin(STRUCT),'struct|','single|')+x.grp))
x['ad']=(x.dn-x.sn).abs()
x['dev_nsend']=x.groupby('device_id').sender_id.transform('nunique')
x['sender_ndev']=df[df.dn!=df.sn].groupby('sender_id').device_id.nunique().reindex(x.sender_id).values
# how many times does device owner use own phone
own=df[df.dn==df.sn].groupby('sender_id').size()
x['owner_rows']=('AC'+x.dn.astype(str).str.zfill(6)).map(own).fillna(0)
pd.set_option('display.width',250)
print(x.groupby('k').agg(n=('ad','size'),ad_med=('ad','median'),ad_lt1000=('ad',lambda s:(s<1000).mean()),dev_nsend=('dev_nsend','mean'),
   owner_rows=('owner_rows','mean'),owner0=('owner_rows',lambda s:(s==0).mean()),devage=('device_age_days','median')).round(3).to_string())
# same device used by sender in other rows?
x['s_dev_rows']=x.groupby(['sender_id','device_id']).txn_id.transform('size')
print(x.groupby('k').s_dev_rows.agg(['mean',lambda s:(s>1).mean()]).round(3))
