import pandas as pd, numpy as np
P='/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/'
df=pd.read_parquet('/tmp/hunt5/raw.parquet')
v10=pd.read_csv(P+'submissions/v10_v6_octrules.csv').rename(columns={'id':'txn_id','label':'pred'})
df=df.merge(v10,on='txn_id',how='left'); df['dn']=df.device_id.str[2:].astype(int)
tr=df[(df.is_test==0)&(df.dn>=420000)&(df.dn<428455)]
# majority non-legit label per device
d=tr[tr.label!='legit'].groupby('dn').label.agg(lambda s:s.value_counts().index[0])
dn=tr.groupby('dn').size()
D=pd.DataFrame({'lab':d,'n':dn}).dropna()
# run-length of classes in device order
D=D.sort_index(); D['run']=(D.lab!=D.lab.shift()).cumsum()
R=D.groupby('run').agg(lab=('lab','first'),d0=('lab',lambda s:s.index.min()),d1=('lab',lambda s:s.index.max()),ndev=('n','size'),rows=('n','sum'))
print(R[R.ndev>=5].to_string())
print(D.groupby('lab').agg(ndev=('n','size'),rows=('n','sum')))
# multi-row share by class in block
print(D.assign(m=D.n>1).groupby('lab').m.mean().round(2))
