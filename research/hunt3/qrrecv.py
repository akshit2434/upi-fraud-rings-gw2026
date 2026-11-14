from base import *
d=load(); d=d[d.grp!='aug'].copy()
pd.set_option('display.width',250,'display.max_columns',30,'display.max_rows',200)
d['lab']=np.where(d.grp=='oct',d.v9,d.label)
q=d[(d.txn_type=='QR')&(d.receiver_type=='P2P')]
R_=q.groupby('grp').receiver_id.apply(set)
for g in ['sep','oct']:
    x=d[(d.grp==g)&d.receiver_id.isin(R_[g])&~((d.txn_type=='QR'))]
    print(g,'non-QR rows into QR-P2P receivers',len(x)); print(pd.crosstab([x.txn_type,x.mis],x.lab))
    y=d[(d.grp==g)&d.sender_id.isin(R_[g])]
    print(g,'rows SENT by QR-P2P receivers',len(y)); print(y.lab.value_counts().to_dict())
# all train: QR->P2P receivers in Jul-Sep; their Oct incoming
tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv',usecols=['receiver_id','receiver_type','txn_type','label'])
qr_recv=set(tr[(tr.txn_type=='QR')&(tr.receiver_type=='P2P')].receiver_id)
o=d[(d.grp=='oct')&d.receiver_id.isin(qr_recv)]; print('Oct rows into train QR-P2P receivers',len(o)); print(pd.crosstab([o.txn_type,o.mis],o.lab))
