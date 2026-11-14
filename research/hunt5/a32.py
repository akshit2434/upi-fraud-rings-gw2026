import pandas as pd, numpy as np
df=pd.read_parquet('/tmp/hunt5/raw.parquet'); df['dn']=df.device_id.str[2:].astype(int); df['sn']=df.sender_id.str[2:].astype(int)
df['grp']=np.where(df.is_test==1,'test',np.where(df.label=='legit','legit',np.where(df.conn,'C','I')))
df['int']=df.amount%1==0; df['r10']=df.amount%10==0
tr=df[df.is_test==0]
print(tr.groupby(['label',tr.conn]).int.mean().unstack().round(3))
print(pd.crosstab([tr.grp],[tr.receiver_type,tr.txn_type],values=tr.int,aggfunc='mean').round(3).to_string())
print(df.groupby('grp').int.mean())
