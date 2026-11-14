import pandas as pd, numpy as np, sys
sys.path.insert(0,'/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/research/hunt')
from load import load
df=load(); tr=df[df.is_test==0]; te=df[df.is_test==1].copy()
sub=pd.read_csv('/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/submissions/v10_v6_octrules.csv').set_index('id').label
te['p']=te.txn_id.map(sub)
nov=te[te.ts>='2026-11-01']; print('Nov test rows',len(nov),nov.p.value_counts().to_dict()); print(nov.txn_type.value_counts().to_dict())
print(nov[nov.p=='legit'][['timestamp','sender_id','receiver_id','device_id','amount','txn_type','merchant_category']].to_string())
# Train: next-month spill rows (Aug rows of Jul... ) are they all fraud? 
for m,nx in [('2026-07','2026-08'),('2026-08','2026-09')]:
    pass
o=tr[(tr.ts>='2026-10-01')&(tr.label!='legit')]
pd.set_option('display.width',250)
for c in ['sender_id','device_id']:
    k=te[te[c].isin(set(o[c]))]
    print(c,len(k),k.p.value_counts().to_dict())
    lk=k[k.p=='legit']; print(lk[['timestamp',c,'receiver_id','amount','txn_type','merchant_category']].head(15).to_string())
    print('their train-Oct labels:',o[o[c].isin(lk[c])].label.value_counts().to_dict())
