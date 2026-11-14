import pandas as pd, numpy as np
P='/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/'
df=pd.read_parquet('/tmp/hunt5/raw.parquet')
v10=pd.read_csv(P+'submissions/v10_v6_octrules.csv').rename(columns={'id':'txn_id','label':'pred'})
df=df.merge(v10,on='txn_id',how='left')
df['dn']=df.device_id.str[2:].astype(int)
df['own']=df.device_id.str[2:]==df.sender_id.str[2:]
tr=df[(df.is_test==0)&(df.dn>=420000)&(df.dn<428455)].sort_values('dn')
te=df[(df.is_test==1)&(df.dn>=473632)&(df.dn<474552)].sort_values('dn')
def runs(x,col,b):
    x=x.assign(bin=(x.dn-x.dn.min())//b)
    return x.groupby('bin').agg(dn0=('dn','min'),n=('txn_id','size'),ndev=('device_id','nunique'),top=(col,lambda s:s.value_counts().head(3).to_dict()))
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',120)
print('TRAIN block by 250-device bins'); print(runs(tr,'label',250).to_string())
print('OCT block by 25-device bins'); print(runs(te,'pred',25).to_string())
print('train block label counts (all months)'); print(tr.label.value_counts())
print('train block per month rows', tr.timestamp.str[:7].value_counts().sort_index().to_dict())
