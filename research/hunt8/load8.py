# hunt8 loader: raw string frame + groups. grp: legit / block / conn / stealth / test ; lp = labelled fraud in oofbadfull_u (OOF P(legit) high)
import pandas as pd, numpy as np, os, sys
sys.path.insert(0,'../hunt6')
R='/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/'
C='/private/tmp/claude-501/-Users-akshit2434--t3-scratch-2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/5cd56af9-d84b-41e7-976e-81ecd813ea6b/scratchpad/'
from load import load, fauc
def raw():
    p=C+'raw8.parquet'
    if os.path.exists(p): return pd.read_parquet(p)
    tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv',dtype=str,keep_default_na=False)
    te=pd.read_csv(R+'data/upi-fraud-rings/test.csv',dtype=str,keep_default_na=False)
    te['label']=''
    r=pd.concat([tr,te],ignore_index=True)
    r.to_parquet(p); return r
def base():
    df=load(); r=raw()
    assert (df.txn_id.values==r.txn_id.values).all()
    bad=pd.read_parquet(R+'data/oofbadfull_u.parquet')
    df['lp']=df.txn_id.isin(set(bad.txn_id))&df.label.notna()&(df.label!='legit')
    df['g']=np.where(df.is_test==1,'test',np.where(df.label=='legit','legit',np.where(df.lp&~df.block,'LP','GEN')))
    return df,r
