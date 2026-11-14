from base import *
tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv'); c=pd.read_parquet(R+'data/conn.parquet'); tr=tr.merge(c,on='txn_id')
pd.set_option('display.width',250,'display.max_columns',30,'display.max_rows',300)
q=tr[tr.label=='qr_swap']; q['mis']=q.qr_vpa_mismatch.map({0:'0',1:'1'}).fillna('NA')
print(pd.crosstab([q.txn_type,q.receiver_type,q.mis],q.conn))
a=tr[(tr.txn_type=='QR')&(tr.receiver_type=='MERCHANT')&tr.qr_vpa_mismatch.isna()&(tr.amount%100!=0)]
print(a[['timestamp','sender_id','receiver_id','amount','merchant_category','label','conn']].to_string())
# qr_swap on QR->MERCHANT mis0: amount round? categories?
z=q[(q.receiver_type=='MERCHANT')&(q.txn_type=='QR')]
print(z.groupby('mis').agg(n=('amount','size'),round=('amount',lambda s:(s%100==0).mean()),conn=('conn','mean')))
print(z.merchant_category.value_counts(normalize=True).round(2).to_dict())
