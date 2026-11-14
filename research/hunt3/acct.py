from base import *
pd.set_option('display.width',250,'display.max_columns',30,'display.max_rows',400)
tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv'); te=pd.read_csv(R+'data/upi-fraud-rings/test.csv')
v9=pd.read_csv(R+'submissions/v9_v6_qrrule.csv').rename(columns={'id':'txn_id','label':'v9'}); te=te.merge(v9,on='txn_id')
for x in (tr,te): x['rn']=x.receiver_id.str[2:].astype(int); x['sn']=x.sender_id.str[2:].astype(int)
p=tr[tr.receiver_type=='P2P']; q=te[te.receiver_type=='P2P']
bins=np.arange(0,420001,10000)
a=p.groupby(pd.cut(p.rn,bins)).agg(n=('label','size'),fr=('label',lambda s:(s!='legit').mean()))
b=q.groupby(pd.cut(q.rn,bins)).agg(nO=('v9','size'),frO=('v9',lambda s:(s!='legit').mean()))
print(a.join(b).round(3).to_string())
s=tr.groupby(pd.cut(tr.sn,bins)).agg(n=('label','size'),fr=('label',lambda s:(s!='legit').mean()))
print(s.round(3).T.to_string())
m=tr[tr.receiver_type=='MERCHANT']; m['mn']=m.receiver_id.str[2:].astype(int)
print(m.groupby(pd.cut(m.mn,np.arange(0,30001,1000))).agg(n=('label','size'),fr=('label',lambda s:(s!='legit').mean())).round(3).T.to_string())
