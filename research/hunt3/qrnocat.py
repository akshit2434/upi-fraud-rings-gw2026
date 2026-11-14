from base import *
tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv'); te=pd.read_csv(R+'data/upi-fraud-rings/test.csv')
v9=pd.read_csv(R+'submissions/v9_v6_qrrule.csv').rename(columns={'id':'txn_id','label':'v9'}); te=te.merge(v9,on='txn_id')
pd.set_option('display.width',250,'display.max_columns',30,'display.max_rows',300)
a=tr[(tr.txn_type=='QR')&(tr.merchant_category.isna())]; b=te[(te.txn_type=='QR')&(te.merchant_category.isna())]
print(pd.crosstab([a.receiver_type,a.qr_vpa_mismatch.fillna(-1)],a.label)); print(pd.crosstab([b.receiver_type,b.qr_vpa_mismatch.fillna(-1)],b.v9))
print(a.amount.describe().round(1).to_dict()); print(b[['amount','v9','new_payee','hour' if 'hour' in b else 'timestamp']].to_string())
