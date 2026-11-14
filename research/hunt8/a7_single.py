# block singleton devices (fraud loop, new phone) vs legit second-phone singletons vs own-phone LP vs own-phone legit
from load8 import *
df,r=base()
acc=pd.read_csv(R+'data/upi-fraud-rings/accounts.csv')
df=df.merge(acc[['account_id','home_city','opened_on']].rename(columns={'account_id':'sender_id'}),on='sender_id',how='left')
df['away']=(df.device_city!=df.home_city).astype(int)
df['hr']=df.ts.dt.hour
df['ndev']=df.groupby('device_id').t.transform('size')
tr=df[df.is_test==0]
blk=(tr.devn>=420000)&(tr.devn<=428454)
sec=(tr.devn>428454)&(tr.devn<473632)
S={'blk_single':tr[blk&(tr.ndev==1)],'sec_single':tr[sec&(tr.ndev==1)],'sec_multi':tr[sec&(tr.ndev>1)],'own_LP':tr[tr.own&(tr.g=='LP')],'own_legit':tr[tr.own&(tr.g=='legit')]}
for k,v in S.items(): print(k,len(v),v.label.value_counts().head(13).to_dict())
rows={}
for k,v in S.items():
    rows[k]={'n':len(v),'merchant':(v.receiver_type=='MERCHANT').mean(),'QR':(v.txn_type=='QR').mean(),'PAY':(v.txn_type=='PAY').mean(),'COLLECT':(v.txn_type=='COLLECT').mean(),'AUTOPAY':(v.txn_type=='AUTOPAY').mean(),
    'amt_med':v.amount.median(),'bal_med':v.balance_before.median(),'sim_med':v.sim_age_hours.median(),'dev_med':v.device_age_days.median(),'dev<2':(v.device_age_days<2).mean(),'sim<72':(v.sim_age_hours<72).mean(),
    'login_med':v.login_to_txn_sec.median(),'pin>0':(v.pin_retries>0).mean(),'newpayee':v.new_payee.mean(),'away':v.away.mean(),'night':v.hr.isin([0,1,2,3,4,5]).mean(),'acct_med':v.sender_account_age_days.median(),
    'ip_med':v.ip_risk_score.median(),'v3':v.app_version.astype(str).str.startswith('3').mean(),'mis1':(v.qr_vpa_mismatch==1).mean(),'misna':v.qr_vpa_mismatch.isna().mean(),'amt_int':(v.amount%1==0).mean()}
print(pd.DataFrame(rows).round(3).to_string())
