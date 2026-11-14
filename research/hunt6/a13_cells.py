# generator-bug scan: structural cells (null patterns x types) with fraud enrichment, LP-only and all
from load import *
df=load(); m=sepval(df)
acc=pd.read_csv(R+'data/upi-fraud-rings/accounts.csv').set_index('account_id')
mer=pd.read_csv(R+'data/upi-fraud-rings/merchants.csv').set_index('merchant_id')
s=df.copy()
s['rpre']=s.receiver_id.str[:2]; s['mcnull']=s.merchant_category.isna(); s['qrs']=s.qr_vpa_mismatch.astype(str)
s['catok']=np.where(s.receiver_type=='MERCHANT',(s.receiver_id.map(mer.category)==s.merchant_category).astype(str),'na')
s['self']=(s.sender_id==s.receiver_id)
s['dcity_known']=s.device_city.isin(set(acc.home_city))
s['cents0']=(s.amount*100%1).round(6)!=0  # more than 2 decimals
s['balneg']=s.balance_before<0; s['amtle0']=s.amount<=0
s['simgtacct']=s.sim_age_hours/24>s.sender_account_age_days+1
s['devgtacct']=s.device_age_days>s.sender_account_age_days+1
s['login0']=s.login_to_txn_sec<=0
cells=['rpre','receiver_type','txn_type','mcnull','qrs','catok','self','dcity_known','cents0','balneg','amtle0','simgtacct','devgtacct','login0']
s=s.merge(m,on='txn_id',how='left')
tr=s[s.is_test==0]; te=s[s.is_test==1]
s['g']=np.where(s.is_test==1,'test',np.where(s.label=='legit','legit',np.where((s.grp=='stealth')&(s.pf<0.05),'LP','fraud_other')))
for c in cells:
    t=pd.crosstab(s[c],s.g)
    if len(t)>1: print(c); print(t)
# multi-key cell
key=s[['rpre','receiver_type','txn_type','mcnull','qrs','catok']].astype(str).agg('|'.join,axis=1)
t=pd.crosstab(key,s.g); print(t.to_string())
