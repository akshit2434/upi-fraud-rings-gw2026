import pandas as pd, numpy as np
P='/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/'
df=pd.read_parquet('/tmp/hunt5/raw.parquet')
v10=pd.read_csv(P+'submissions/v10_v6_octrules.csv').rename(columns={'id':'txn_id','label':'pred'})
te=df[df.is_test==1].merge(v10,on='txn_id')
sep=df[(df.is_test==0)&(df.timestamp>='2026-09-01')&(df.timestamp<'2026-10-01')]
rng=np.random.default_rng(0); sep51=sep[rng.random(len(sep))<0.51]
def fanin(x,name):
    p=x[x.receiver_type=='P2P']
    g=p.groupby('receiver_id').sender_id.nunique()
    p=p.assign(fi=p.receiver_id.map(g))
    out=p.groupby(pd.cut(p.fi,[0,1,2,3,4,6,10,20,1000])).agg(rows=('txn_id','size'),recv=('receiver_id','nunique'))
    if 'pred' in p: out['flag']=p.groupby(pd.cut(p.fi,[0,1,2,3,4,6,10,20,1000])).pred.apply(lambda s:(s!='legit').mean())
    if 'label' in p and p.label.notna().any(): out['fr']=p.groupby(pd.cut(p.fi,[0,1,2,3,4,6,10,20,1000])).label.apply(lambda s:(s!='legit').mean())
    print(name, len(x)); print(out.round(3))
fanin(sep51,'sep51'); fanin(te,'oct')
# out-degree of senders
for x,n in [(sep51,'sep51'),(te,'oct')]:
    g=x.groupby('sender_id').receiver_id.nunique(); print(n,'sender distinct recv dist', g.value_counts().sort_index().head(12).to_dict())
    g=x.groupby('device_id').sender_id.nunique(); print(n,'device distinct senders', g.value_counts().sort_index().head(12).to_dict())
