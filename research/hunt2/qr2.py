from load import *
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv'); te=pd.read_csv(R+'data/upi-fraud-rings/test.csv')
sub=pd.read_csv(R+'submissions/v6_R1_bag.csv').rename(columns={'id':'txn_id','label':'v6'}); te=te.merge(sub,on='txn_id')
P=np.load(R+'preds/v6_R1_bag_test_P.npy'); ids=pd.read_csv(R+'preds/v6_R1_bag_test_ids.csv').iloc[:,0].values
te=te.merge(pd.DataFrame({'txn_id':ids,'pqr':P[:,7],'pfr':1-P[:,0]}),on='txn_id')
def prof(x):
    ts=pd.to_datetime(x.timestamp); h=ts.dt.hour
    return pd.Series({'n':len(x),'amt_med':x.amount.median(),'round100':(x.amount%100==0).mean(),'newp':x.new_payee.mean(),
      'hr_med':h.median(),'night':(h<6).mean(),'sim_med':x.sim_age_hours.median(),'devage_med':x.device_age_days.median(),
      'own':(x.device_id.str[3:]==x.sender_id.str[2:]).mean(),'pin':(x.pin_retries>0).mean(),'login_med':x.login_to_txn_sec.median(),
      'ip_med':x.ip_risk_score.median(),'amt_bal':(x.amount/x.balance_before).median(),'cat_na':x.merchant_category.isna().mean(),
      'app_A':(x.app=='APP_A').mean()})
q=lambda x:x[(x.txn_type=='QR')&(x.receiver_type=='P2P')]
rows={}
t=q(tr)
for lab in ['qr_swap','legit']:
    for m in [0,1]: rows[f'train {lab} mis={m}']=prof(t[(t.label==lab)&(t.qr_vpa_mismatch==m)])
o=q(te)
for m in [0,1]: rows[f'test mis={m}']=prof(o[o.qr_vpa_mismatch==m])
rows['test mis=NaN']=prof(o[o.qr_vpa_mismatch.isna()])
rows['test QR merch (ref)']=prof(te[(te.txn_type=='QR')&(te.receiver_type=='MERCHANT')])
print(pd.DataFrame(rows).T.round(3).to_string())
n=o[o.qr_vpa_mismatch.isna()]
print('null rows v6 pqr quantiles',n.pqr.quantile([.5,.9,.99]).round(4).to_dict(),'pfr',n.pfr.quantile([.5,.9,.99]).round(4).to_dict())
print('receivers distinct',n.receiver_id.nunique(),'senders',n.sender_id.nunique(),'dates',pd.to_datetime(n.timestamp).dt.day.value_counts().sort_index().to_dict())
print('null QR by app', n.app.value_counts().to_dict(), 'app_version', n.app_version.value_counts().to_dict())
# nulls by txn_type receiver_type in train vs test
for nm,x in [('train',tr),('test',te)]:
    print(nm, x.assign(nul=x.qr_vpa_mismatch.isna()).groupby(['txn_type','receiver_type']).nul.agg(['mean','size']).round(3).to_dict('index'))
# receivers of null rows: do they appear elsewhere in test / train?
rv=set(n.receiver_id); print('null receivers seen in train', len(rv & set(tr.receiver_id)), 'as receiver of other test rows', te[te.receiver_id.isin(rv)&~te.txn_id.isin(n.txn_id)].shape[0])
print(n[['timestamp','sender_id','receiver_id','amount','new_payee','sim_age_hours','device_age_days','login_to_txn_sec','pin_retries']].head(12))
