import pandas as pd, numpy as np
df=pd.read_parquet('/tmp/hunt5/raw.parquet'); df['dn']=df.device_id.str[2:].astype(int)
tr=df[(df.is_test==0)&(df.dn>=420000)&(df.dn<428455)].copy()
zt=tr[tr.label!='legit'].groupby('dn').label.agg(lambda s:s.value_counts().index[0]); tr['zone']=tr.dn.map(zt)
tr['collike']=(tr.receiver_type=='MERCHANT')&(tr.amount%500==0)&(tr.txn_type!='REFUND')
print(pd.crosstab(tr.zone,[tr.collike]).to_string())
print(tr[tr.collike].groupby('zone').label.value_counts().unstack(fill_value=0).to_string())
# whole-train: collusion-like rows on own phones labels
a=df[(df.is_test==0)&(df.receiver_type=='MERCHANT')&(df.amount%500==0)&(df.txn_type!='REFUND')]
print(a.label.value_counts().head(6))
