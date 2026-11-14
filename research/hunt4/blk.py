import pandas as pd, numpy as np, sys
sys.path.insert(0,'/tmp/hunt3')
from base import sep_pred, R
tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv',usecols=['txn_id','timestamp','device_id','sender_id','label','txn_type','qr_vpa_mismatch'])
sp=sep_pred(); s=tr.merge(sp,on='txn_id')
s['dn']=s.device_id.str[2:].astype(int)
b=s[(s.dn>=420000)&(s.dn<428455)]
print('Sep(51% frame) block rows',len(b),'pred fraud',(b.sp!='legit').sum(),'pred legit',(b.sp=='legit').sum(),' true fraud among pred-legit',round((b[b.sp=='legit'].label!='legit').mean(),3))
print('all Sep51 frame rows',len(s),'pred fraud',(s.sp!='legit').sum(), 'true fraud',(s.label!='legit').sum(),'marked nonQR',((s.txn_type!='QR')&s.qr_vpa_mismatch.notna()).sum())
print('Oct test: block 564 rows, v9 pred legit 212 / fraud 352; all test pred fraud (v9)',)
