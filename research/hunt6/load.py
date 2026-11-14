import pandas as pd, numpy as np, os
R='/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/'
C='/private/tmp/claude-501/-Users-akshit2434--t3-scratch-2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/5cd56af9-d84b-41e7-976e-81ecd813ea6b/scratchpad/'
os.makedirs(C,exist_ok=True)
def load():
    p=C+'all6.parquet'
    if os.path.exists(p): return pd.read_parquet(p)
    tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv'); te=pd.read_csv(R+'data/upi-fraud-rings/test.csv')
    tr['is_test']=0; te['is_test']=1; te['label']=None
    df=pd.concat([tr,te],ignore_index=True)
    df['ts']=pd.to_datetime(df.timestamp)
    df['t']=(df.ts-pd.Timestamp('2026-07-01')).dt.total_seconds()
    c=pd.read_parquet(R+'data/conn.parquet'); df=df.merge(c,on='txn_id',how='left'); df['conn']=df.conn.fillna(False)
    dn=pd.to_numeric(df.device_id.str[2:],errors='coerce')
    df['devn']=dn
    df['block']=((dn>=420000)&(dn<=428454))|((dn>=473632)&(dn<=474551))
    df['own']=(dn==pd.to_numeric(df.sender_id.str[2:]))
    df['mon']=df.ts.dt.month
    fr=df.label.notna()&(df.label!='legit')
    df['grp']=np.where(df.is_test==1,'test',np.where(df.label=='legit','legit',np.where(df.block,'block',np.where(df.conn,'conn','stealth'))))
    df.to_parquet(p); return df
from scipy.stats import rankdata
def fauc(y,s):
    y=np.asarray(y,bool); s=np.asarray(s,float); m=~np.isnan(s); y=y[m]; s=s[m]
    if y.sum()==0 or (~y).sum()==0: return np.nan
    r=rankdata(s); n1=y.sum(); n0=len(y)-n1
    return (r[y].sum()-n1*(n1+1)/2)/(n1*n0)
def sepval(df):
    P=np.load('/tmp/P_bag4_R1.npy'); i=np.load('/tmp/P_bag4_R1_ids.npy',allow_pickle=True)
    m=pd.DataFrame({'txn_id':i,'pf':1-P[:,0]})
    return m
