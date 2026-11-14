from base import *
tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv'); te=pd.read_csv(R+'data/upi-fraud-rings/test.csv')
v9=pd.read_csv(R+'submissions/v9_v6_qrrule.csv').rename(columns={'id':'txn_id','label':'v9'}); te=te.merge(v9,on='txn_id')
for x in (tr,te):
    x['mis']=x.qr_vpa_mismatch.map({0.0:'0',1.0:'1'}).fillna('NA'); x['catn']=np.where(x.merchant_category.isna(),'nocat','cat')
    x['mon']=x.timestamp.str[:7]
tr=tr[tr.mon<'2026-10']
k=['txn_type','receiver_type','mis','catn']
a=tr.groupby(k).agg(n_tr=('label','size'),fr=('label',lambda s:(s!='legit').mean()),top=('label',lambda s:s.value_counts().head(3).round(2).to_dict() if False else (s.value_counts(normalize=True).head(3).round(2).to_dict())))
sep=tr[tr.mon=='2026-09'].groupby(k).size().rename('sep51')*0.51
b=te.groupby(k).agg(n_te=('v9','size'),v9fr=('v9',lambda s:(s!='legit').mean()))
t=a.join(sep,how='outer').join(b,how='outer').fillna(0)
t['ratio']=t.n_te/t.sep51.replace(0,np.nan)
pd.set_option('display.width',250,'display.max_colwidth',80,'display.max_rows',200)
t.to_pickle("struct.pkl"); print(t.drop(columns="top").sort_values('n_tr',ascending=False))
# category by receiver type


for i,r in t.sort_values('n_tr',ascending=False).iterrows(): print(i, r['top'])
# v9 label distribution in test anomalous cells
an=te[~(((te.txn_type=='QR')&(te.mis!='NA'))|((te.txn_type!='QR')&(te.mis=='NA')))]
print(pd.crosstab([an.txn_type,an.receiver_type,an.mis],an.v9))
ant=tr[~(((tr.txn_type=='QR')&(tr.mis!='NA'))|((tr.txn_type!='QR')&(tr.mis=='NA')))]
print(pd.crosstab([ant.txn_type,ant.receiver_type,ant.mis],ant.label))
