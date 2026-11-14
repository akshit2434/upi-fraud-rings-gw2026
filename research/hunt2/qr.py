from load import *
pd.set_option('display.width',250)
d=load(); d=d[~((d._is_test==0)&(~d._month.isin(['2026-08','2026-09'])))]
q=d[(d.txn_type=='QR')]
q=q.assign(mis=q.qr_vpa_mismatch.fillna(-1).astype(int))
print(q.groupby(['grp','receiver_type','mis']).size().unstack([0]).fillna(0))
for g in ['aug','sep']:
    s=q[(q.grp==g)]
    print(g, pd.crosstab([s.receiver_type,s.mis],s.label))
o=q[q.grp=='oct']; print(pd.crosstab([o.receiver_type,o.mis],o.v6))
print('-----')
for g in ['aug','sep']:
    s=q[(q.grp==g)]
    print(g); print(pd.crosstab([s.receiver_type,s.mis],s.label.where(s.label.isin(['legit','qr_swap']),'other')))
print(pd.crosstab([o.receiver_type,o.mis],o.v6.where(o.v6.isin(['legit','qr_swap']),'other')))
# all of train
tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv'); te=pd.read_csv(R+'data/upi-fraud-rings/test.csv')
for nm,x in [('train',tr),('test',te)]:
    x=x[x.txn_type=='QR']; x=x.assign(mis=x.qr_vpa_mismatch.fillna(-1).astype(int))
    print(nm, x.groupby(['receiver_type','mis']).size())
qq=tr[(tr.txn_type=='QR')&(tr.receiver_type=='P2P')]
print('train QR->P2P labels', qq.label.value_counts().to_dict())
