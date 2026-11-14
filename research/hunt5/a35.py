import pandas as pd, numpy as np
df=pd.read_parquet('/tmp/hunt5/raw.parquet'); df['dn']=df.device_id.str[2:].astype(int); df['sn']=df.sender_id.str[2:].astype(int)
tr=df[(df.is_test==0)&(df.label!='legit')].sort_values(['sender_id','t']).copy()
tr['blk']=(tr.dn>=420000)&(tr.dn<428455)
g=tr.groupby('sender_id'); tr['gap']=tr.t-g.t.shift(1); tr['new']=(tr.gap.isna())|(tr.gap>1800)
tr['ev']=tr.new.cumsum()
e=tr.groupby('ev').agg(n=('label','size'),nl=('label','nunique'),labs=('label',lambda s:'/'.join(sorted(set(s)))),blk=('blk','mean'),first=('label','first'))
e=e[e.n>=2]
print('sender bursts (<=30m, all fraud-labelled rows) n>=2:',len(e))
print('mixed share by zone: block %.3f  non-block %.3f'%(e[e.blk==1].nl.gt(1).mean(), e[e.blk==0].nl.gt(1).mean()))
print(e[e.nl>1].labs.value_counts().head(12))
print(e[e.nl==1].labs.value_counts().head(12))
# row-level: if row-level swap noise rate q between sim/ATO, P(2-row event mixed)=2q(1-q)
x=e[(e.blk==0)&e.labs.str.contains('sim_swap|account_takeover')&(e.n==2)]
print('non-block 2-row sim/ATO events',len(x),'mixed',(x.nl>1).mean().round(3))
x=e[(e.blk==1)&e.labs.str.contains('sim_swap|account_takeover')&(e.n==2)]
print('block 2-row sim/ATO events',len(x),'mixed',(x.nl>1).mean().round(3))
