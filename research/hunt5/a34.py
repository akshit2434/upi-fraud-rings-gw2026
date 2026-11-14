import pandas as pd, numpy as np, sys
sys.path.insert(0,'/tmp/hunt3'); from base import sep_pred
R='/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/'
sys.path.insert(0,R+'src'); from common import macro_f1
df=pd.read_parquet('/tmp/hunt5/raw.parquet')
s=df[['txn_id','label']].merge(sep_pred(),on='txn_id')
LAB=['legit','mule_account','velocity_bot','layering_chain','sim_swap','account_takeover','phishing_collect','qr_swap','refund_abuse','fake_cashback','loan_app_extortion','merchant_collusion']
from sklearn.metrics import f1_score
f=lambda y,p: f1_score(y,p,labels=LAB,average='macro')
print('v6 Sep', round(f(s.label,s.sp),4))
o=s.sp.copy(); m=(s.sp!='legit')&(s.label!='legit'); o[m]=s.label[m]; print('oracle typing on flagged fraud', round(f(s.label,o),4))
for pair in [('account_takeover','sim_swap'),('fake_cashback','phishing_collect'),('velocity_bot','mule_account'),('mule_account','layering_chain'),('refund_abuse','qr_swap')]:
    o=s.sp.copy(); mm=m&s.label.isin(pair)&s.sp.isin(pair); o[mm]=s.label[mm]; print(' fix only',pair, round(f(s.label,o)-f(s.label,s.sp),4))
print(pd.Series(f1_score(s.label,s.sp,labels=LAB,average=None),index=LAB).round(3).to_dict())
