import pandas as pd, numpy as np, os
R='/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/'
STRUCT=['mule_account','velocity_bot','layering_chain','sim_swap','account_takeover','merchant_collusion']
def load():
    p='/tmp/hunt/all.parquet'
    if os.path.exists(p): return pd.read_parquet(p)
    tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv'); te=pd.read_csv(R+'data/upi-fraud-rings/test.csv')
    tr['is_test']=0; te['is_test']=1; te['label']=None
    df=pd.concat([tr,te],ignore_index=True)
    df['ts']=pd.to_datetime(df.timestamp)
    df['t']=(df.ts-pd.Timestamp('2026-07-01')).dt.total_seconds()
    c=pd.read_parquet(R+'data/conn.parquet'); df=df.merge(c,on='txn_id',how='left')
    df['grp']=np.where(df.label=='legit','legit',np.where(df.label.isna(),'test',np.where(df.conn.fillna(False),'conn','iso')))
    df.to_parquet(p); return df
def auc(y,s):
    from sklearn.metrics import roc_auc_score
    m=~pd.isna(s); return roc_auc_score(y[m],s[m]) if m.sum()>0 else np.nan
from scipy.stats import rankdata
def fauc(y,s):
    y=np.asarray(y,bool); s=np.asarray(s,float); m=~np.isnan(s); y=y[m]; s=s[m]
    if y.sum()==0 or (~y).sum()==0: return np.nan
    r=rankdata(s); n1=y.sum(); n0=len(y)-n1
    return (r[y].sum()-n1*(n1+1)/2)/(n1*n0)
