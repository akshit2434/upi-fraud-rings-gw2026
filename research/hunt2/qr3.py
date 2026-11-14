from load import *
tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv'); te=pd.read_csv(R+'data/upi-fraud-rings/test.csv')
sub=pd.read_csv(R+'submissions/v6_R1_bag.csv').rename(columns={'id':'txn_id','label':'v6'}); te=te.merge(sub,on='txn_id')
qp=lambda x:(x.txn_type=='QR')&(x.receiver_type=='P2P')
n=te[qp(te)&te.qr_vpa_mismatch.isna()]; c=te[qp(te)&te.qr_vpa_mismatch.notna()]
rv=set(n.receiver_id)
t=tr[tr.receiver_id.isin(rv)]
print('train rows to null-QR receivers:',len(t),'labels',t.label.value_counts().head(6).to_dict())
print('train qr_swap rows to them', (t.label=='qr_swap').sum(), 'receivers with any train qr_swap', t[t.label=='qr_swap'].receiver_id.nunique(),'of',len(rv))
# baseline: random P2P receivers
import random
allp2p=te[te.receiver_type=='P2P'].receiver_id.drop_duplicates().sample(140,random_state=0)
tb=tr[tr.receiver_id.isin(set(allp2p))]; print('baseline random P2P receivers: train rows',len(tb),'qr_swap',(tb.label=='qr_swap').sum(), 'fraud share %.3f'%(tb.label!='legit').mean())
cr=set(c.receiver_id); tc=tr[tr.receiver_id.isin(cr)]; print('checked-QR-P2P test receivers: train rows',len(tc),'qr_swap',(tc.label=='qr_swap').sum(),'fraud share %.3f'%(tc.label!='legit').mean())
print('fraud share of train rows to null receivers %.3f'%(t.label!='legit').mean())
# train qr_swap P2P receivers: how often do they receive other QR
q=tr[qp(tr)]; print('train QR->P2P receivers distinct',q.receiver_id.nunique(),'rows',len(q))
print('null rows overlap receivers with checked test QR->P2P', len(rv&cr))
# timing by day of month for null vs checked in test
for nm,x in [('null',n),('checked',c)]:
    print(nm, pd.to_datetime(x.timestamp).dt.day.describe()[['mean','25%','75%']].round(1).to_dict())
