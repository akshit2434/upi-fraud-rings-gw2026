import pandas as pd, numpy as np, sys
sys.path.insert(0,'/tmp/hunt3')
from base import sep_pred, R
cols=['txn_id','txn_type','qr_vpa_mismatch','receiver_type','device_id','sender_id']
tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv',usecols=cols+['label']); te=pd.read_csv(R+'data/upi-fraud-rings/test.csv',usecols=cols)
s=tr.merge(sep_pred(),on='txn_id')
v9=pd.read_csv(R+'submissions/v9_v6_qrrule.csv').rename(columns={'id':'txn_id','label':'sp'}); t=te.merge(v9,on='txn_id')
for nm,d in [('Sep51 v6',s),('Oct v9',t)]:
    d['mk']=(d.txn_type!='QR')&d.qr_vpa_mismatch.notna(); d['nq']=d.txn_type!='QR'
    f=d[(d.sp!='legit')&d.nq]
    print(nm,'flagged nonQR',len(f),'marked share',round(f.mk.mean(),3))
    print('  by class', f.groupby('sp').mk.mean().round(2).to_dict())
    print('  flagged count by class', d[d.sp!='legit'].sp.value_counts().to_dict())
