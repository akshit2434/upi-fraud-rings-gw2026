from base import *
tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv'); te=pd.read_csv(R+'data/upi-fraud-rings/test.csv')
v9=pd.read_csv(R+'submissions/v9_v6_qrrule.csv').rename(columns={'id':'txn_id','label':'v9'}); te=te.merge(v9,on='txn_id')
pd.set_option('display.width',250,'display.max_columns',30,'display.max_rows',300)
paise=lambda s: (s.round(2)!=s.round(0))
A=te[(te.txn_type=='QR')&(te.receiver_type=='P2P')&te.qr_vpa_mismatch.isna()]
B=te[(te.txn_type=='QR')&(te.receiver_type=='MERCHANT')&te.qr_vpa_mismatch.isna()&(te.amount%100!=0)]
C=tr[(tr.txn_type=='QR')&(tr.receiver_type=='P2P')]
D=tr[(tr.txn_type=='QR')&(tr.receiver_type=='MERCHANT')&(tr.label=='legit')]
for n,x in [('test QRP2P blank',A),('test QRM blank nonround',B),('train QRP2P',C),('train QRM legit',D),('train QRM qr_swap',tr[(tr.txn_type=='QR')&(tr.receiver_type=='MERCHANT')&(tr.label=='qr_swap')])]:
    print(n,len(x),'paise',round(paise(x.amount).mean(),2),'round100',round((x.amount%100==0).mean(),2),'amt med',x.amount.median(),'newp',round(x.new_payee.mean(),2),
          'cat',x.merchant_category.value_counts(normalize=True).round(2).head(4).to_dict())
# merchant receivers of B: train label history
h=tr[tr.receiver_id.isin(B.receiver_id)]
print('train rows to B merchants',len(h),h.label.value_counts().to_dict())
h2=tr[tr.receiver_id.isin(tr[tr.label=='qr_swap'].receiver_id)&(tr.receiver_type=='MERCHANT')]
print('B merchants with qr_swap history', B.receiver_id.isin(tr[tr.label=='qr_swap'].receiver_id).sum(), 'base rate of merchants w/ qr hist among test QR->M rows', te[(te.txn_type=='QR')&(te.receiver_type=='MERCHANT')].receiver_id.isin(tr[tr.label=='qr_swap'].receiver_id).mean())
