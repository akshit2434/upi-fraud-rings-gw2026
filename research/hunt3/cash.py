from base import *
d=load(); d=d[d.grp!='aug']
pd.set_option('display.width',250,'display.max_columns',30,'display.max_rows',200)
d['lab']=np.where(d.grp=='oct',d.v9,d.label)
x=d[(d.amount<=10)]
print(pd.crosstab([x.grp,x.receiver_type,x.txn_type],x.lab))
# amounts <=10 legit vs fraud integer?
x=d[(d.amount<=10)&(d.receiver_type=='P2P')]
print(x.groupby(['grp',x.amount.round(0)==x.amount]).lab.value_counts().unstack().fillna(0))
tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv')
c=tr[tr.label=='fake_cashback']; print('train cashback by type/amt<=10', pd.crosstab(c.txn_type,c.amount<=10))
print(c.amount.describe())
