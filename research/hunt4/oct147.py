import pandas as pd, numpy as np, sys
sys.path.insert(0,'/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/research/hunt')
from load import load
df=load(); tr=df[df.is_test==0]; te=df[df.is_test==1]
o=tr[tr.ts>='2026-10-01']
print(len(o), o.label.value_counts().to_dict()); print(o.ts.min(), o.ts.max())
print('Sep-30 last day train rows', (tr.ts.dt.date.astype(str)=='2026-09-30').sum())
for c in ['sender_id','receiver_id','device_id']:
    s=set(o[c]); m=te[c].isin(s); print(c,'test rows sharing',m.sum(), 'distinct', te[c][m].nunique())
# label of preceding train rows of same sender
pd.set_option('display.width',250)
print(o.sort_values('ts')[['timestamp','sender_id','receiver_id','device_id','amount','txn_type','merchant_category','label']].head(25).to_string())
# Test rows dated Nov
n=te[te.ts>='2026-11-01']; print('nov test',len(n)); print(n[['timestamp','sender_id','receiver_id','device_id','amount','txn_type','merchant_category']].head(20).to_string())
# last train date in Sep per label: fraud tails crossing month boundaries?
