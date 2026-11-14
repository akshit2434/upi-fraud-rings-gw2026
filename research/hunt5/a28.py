import pandas as pd, numpy as np
D='/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/data/upi-fraud-rings/'
df=pd.read_parquet('/tmp/hunt5/rawj.parquet'); df['dn']=df.device_id.str[2:].astype(int); df['sn']=df.sender_id.str[2:].astype(int)
acc=pd.read_csv(D+'accounts.csv'); acc['an']=acc.account_id.str[2:].astype(int)
x=df[(df.dn!=df.sn)&(df.dn<420000)].copy()
x=x.merge(acc[['an','home_city','bank_code','os','kyc_level']].add_prefix('o_'),left_on='dn',right_on='o_an',how='left')
x['samecity']=x.o_home_city==x.s_home_city; x['samebank']=x.o_bank_code==x.s_bank_code; x['sameos']=x.o_os==x.s_os
x['dev_owner_city']=x.device_city==x.o_home_city
x['owner_is_recv']=x.dn==x.receiver_id.str[2:].where(x.receiver_type=='P2P').astype(float)
x['gap']=(x.dn-x.sn).abs()
tr=x[x.is_test==0]
print(len(x),'rows on other accounts phones; test',(x.is_test==1).sum())
print(tr.assign(f=np.where(tr.label=='legit','legit',np.where(tr.conn,'C','I'))).groupby('f')[['samecity','samebank','sameos','dev_owner_city','owner_is_recv']].mean().round(3))
print(tr.groupby('label')[['samecity','sameos','owner_is_recv']].mean().round(3))
print(tr.groupby(tr.label=='legit').gap.describe())
