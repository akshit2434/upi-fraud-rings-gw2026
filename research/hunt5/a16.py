import pandas as pd, numpy as np, sys
sys.path.insert(0,'/tmp/hunt3'); from base import sep_pred
df=pd.read_parquet('/tmp/hunt5/raw.parquet')
df['dn']=df.device_id.str[2:].astype(int); df['sn']=df.sender_id.str[2:].astype(int)
df['own']=df.dn==df.sn
df['blk']=(df.dn>=420000)&(df.dn<428455)
s=df.merge(sep_pred(),on='txn_id')
B=s[s.blk&(s.label!='legit')]
pd.set_option('display.width',250)
print('Sep block fraud rows',len(B)); print(pd.crosstab(B.label,B.sp).to_string())
NB=s[(~s.blk)&(s.label!='legit')&(s.sp!='legit')]
print('Sep non-block flagged fraud'); print(pd.crosstab(NB.label,NB.sp).to_string())
tr=df[df.is_test==0]
print('zone by label (own / block / other)'); print(pd.crosstab(tr.label,np.select([tr.own,tr.blk],['own','blk'],'other')).to_string())
