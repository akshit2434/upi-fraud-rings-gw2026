from base import *
tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv'); c=pd.read_parquet(R+'data/conn.parquet'); tr=tr.merge(c,on='txn_id')
pd.set_option('display.width',250,'display.max_columns',30,'display.max_rows',300)
L=tr[tr.label=='loan_app_extortion']; L['hr']=L.timestamp.str[11:13].astype(int)
print(pd.crosstab([L.receiver_type,L.merchant_category.fillna('NA')],L.conn))
print(pd.crosstab(L.hr,L.conn).T)
print(L.groupby('conn').agg(amt=('amount','median'),np_=('new_payee','mean'),int_=('amount',lambda s:(s.round(0)==s).mean())))
# loan: per sender count of loan rows
print(L.groupby('sender_id').size().value_counts().sort_index().to_dict())
d=load(); d=d[d.grp!='aug'].copy(); d['lab']=np.where(d.grp=='oct',d.v9,d.label)
eve=d.hr.isin([20,21,22,23])
# repeated pair payments in evening, P2P
x=d[eve&(d.receiver_type=='P2P')]
c2=x.groupby(['grp','sender_id','receiver_id']).size().rename('np').reset_index(); x=x.merge(c2,on=['grp','sender_id','receiver_id'])
print(x.groupby(['grp',x.np.clip(upper=4)]).agg(n=('lab','size'),fr=('lab',lambda s:(s!='legit').mean()),loan=('lab',lambda s:(s=='loan_app_extortion').mean())).unstack(0).round(3))
# sender with >=3 evening payments (any receiver) in 7 days
