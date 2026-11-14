from base import *
tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv',usecols=['txn_id','timestamp','sender_id','receiver_id','new_payee','label','txn_type'])
te=pd.read_csv(R+'data/upi-fraud-rings/test.csv',usecols=['txn_id','timestamp','sender_id','receiver_id','new_payee','txn_type'])
v9=pd.read_csv(R+'submissions/v9_v6_qrrule.csv').rename(columns={'id':'txn_id','label':'v9'}); te=te.merge(v9,on='txn_id')
a=pd.concat([tr.assign(src='tr'),te.assign(src='te',label=te.v9)]).sort_values('timestamp')
a['prior']=a.groupby(['sender_id','receiver_id']).cumcount()
a['fraud']=a.label!='legit'
a['mon']=a.timestamp.str[:7]
x=a[a.mon.isin(['2026-09','2026-10'])]
print(x.groupby(['src','new_payee',x.prior.clip(upper=2)]).agg(n=('fraud','size'),fr=('fraud','mean')).round(3))
