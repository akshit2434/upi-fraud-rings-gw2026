from base import *
tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv'); te=pd.read_csv(R+'data/upi-fraud-rings/test.csv')
v9=pd.read_csv(R+'submissions/v9_v6_qrrule.csv').rename(columns={'id':'txn_id','label':'v9'}); te=te.merge(v9,on='txn_id')
P=np.load(R+'preds/v6_R1_bag_test_P.npy'); ids=pd.read_csv(R+'preds/v6_R1_bag_test_ids.csv').iloc[:,0].values
te=te.merge(pd.DataFrame({'txn_id':ids,'pf':1-P[:,0]}),on='txn_id')
pd.set_option('display.width',250,'display.max_columns',30,'display.max_rows',300)
a=tr[(tr.txn_type=='QR')&(tr.receiver_type=='MERCHANT')&tr.qr_vpa_mismatch.isna()]
b=te[(te.txn_type=='QR')&(te.receiver_type=='MERCHANT')&te.qr_vpa_mismatch.isna()]
print('train',len(a),a.timestamp.str[:7].value_counts().sort_index().to_dict()); print('test',len(b))
print(pd.crosstab(a.merchant_category,a.label)); print(pd.crosstab(b.merchant_category,b.v9))
print('train legit amounts', a[a.label=='legit'].amount.describe().round(1).to_dict())
print(b[b.v9=='legit'][['timestamp','sender_id','receiver_id','amount','merchant_category','new_payee','pf','sim_age_hours','device_age_days']].to_string())
a['round']=(a.amount%100==0); b['round']=(b.amount%100==0)
print(pd.crosstab(a.label,a['round'])); print(pd.crosstab(b.v9,b['round']))
