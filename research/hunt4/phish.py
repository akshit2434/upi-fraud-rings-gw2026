import pandas as pd, numpy as np, sys
sys.path.insert(0,'/tmp/hunt3'); sys.path.insert(0,'/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/research/hunt')
from load import load, fauc
from base import sep_pred
df=load(); tr=df[df.is_test==0].copy(); tr['v3']=tr.app_version.astype(str).str.startswith('3')
sp=sep_pred(); tr=tr.merge(sp,on='txn_id',how='left')
for c in ['phishing_collect','qr_swap','refund_abuse','loan_app_extortion','fake_cashback']:
    s=tr[tr.sp.notna()]
    for g in ['conn','iso']:
        x=s[(s.label==c)&(s.grp==g)]
        print(c,g,len(x),'recall all',round((x.sp==c).mean(),3),'recall v3',round((x[x.v3].sp==c).mean(),3),'recall v4',round((x[~x.v3].sp==c).mean(),3),'v3 share',round(x.v3.mean(),3))
# missed iso phishing: profile vs legit
s=tr[tr.sp.notna()]
L=s[s.label=='legit']
for c in ['phishing_collect','qr_swap']:
    x=s[(s.label==c)&(s.grp=='iso')]
    miss=x[x.sp!=c]; hit=x[x.sp==c]
    print('\n',c,'txn_type miss/hit/legit');print(pd.concat([miss.txn_type.value_counts(normalize=True).rename('miss'),hit.txn_type.value_counts(normalize=True).rename('hit'),L.txn_type.value_counts(normalize=True).rename('legit')],axis=1).round(3).T.to_string())
    print(pd.concat([miss.receiver_type.value_counts(normalize=True).rename('miss'),L.receiver_type.value_counts(normalize=True).rename('legit')],axis=1).round(3).T.to_string())
    print('miss v3',round(miss.v3.mean(),3),'newpayee miss/legit',round(miss.new_payee.mean(),3),round(L.new_payee.mean(),3),'cents0',round(((miss.amount*100).round()%100==0).mean(),3),round(((L.amount*100).round()%100==0).mean(),3))
