import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
D='/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/data/upi-fraud-rings/'
df=pd.read_parquet('/tmp/hunt5/raw.parquet')
acc=pd.read_csv(D+'accounts.csv'); mer=pd.read_csv(D+'merchants.csv')
df['rec_cnt']=df.groupby('receiver_id').txn_id.transform('size')
df['sen_cnt']=df.groupby('sender_id').txn_id.transform('size')
df=df.merge(acc.add_prefix('s_'),left_on='sender_id',right_on='s_account_id',how='left')
df=df.merge(mer.add_prefix('m_'),left_on='receiver_id',right_on='m_merchant_id',how='left')
df=df.merge(acc.add_prefix('r_'),left_on='receiver_id',right_on='r_account_id',how='left')
df['m_samecity_home']=(df.m_city==df.s_home_city)
df['m_samecity_dev']=(df.m_city==df.device_city)
df['dev_home']=(df.device_city==df.s_home_city)
df['r_samecity']=(df.r_home_city==df.s_home_city)
df['r_samebank']=(df.r_bank_code==df.s_bank_code)
df.to_parquet('/tmp/hunt5/rawj.parquet')
tr=df[df.is_test==0]
tr=tr.assign(grp=np.where(tr.label=='legit','legit',np.where(tr.conn,'C','I')))
M=tr[tr.receiver_type=='MERCHANT']; P=tr[tr.receiver_type=='P2P']
print('MERCHANT rows'); print(M.groupby('grp')[['m_samecity_home','m_samecity_dev','dev_home']].mean().round(3))
print(M.groupby('label')[['m_samecity_home','m_samecity_dev','dev_home']].mean().round(3))
print('P2P rows'); print(P.groupby('grp')[['r_samecity','r_samebank','dev_home']].mean().round(3))
print(P.groupby('label')[['r_samecity','r_samebank','dev_home']].mean().round(3))
for lab in ['mule_account','velocity_bot','sim_swap','account_takeover','layering_chain','qr_swap','phishing_collect']:
    I=M[(M.label==lab)&(~M.conn)]; L=M[M.label=='legit'].sample(200000,random_state=0)
    y=np.r_[np.ones(len(I)),np.zeros(len(L))]; x=np.r_[I.rec_cnt,L.rec_cnt]
    print(lab,'merchant cnt AUC (low=fraud)',round(1-roc_auc_score(y,x),3), len(I))
