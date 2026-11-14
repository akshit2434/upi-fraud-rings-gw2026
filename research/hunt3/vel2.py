from base import *
d=load(); d=d[d.grp!='aug'].copy()
pd.set_option('display.width',250,'display.max_columns',30,'display.max_rows',200)
d['lab']=np.where(d.grp=='oct',d.v9,d.label); d['hr10']=d.ts.dt.floor('10min')
# device-hour burst: >=3 rows within the same hour on device
d['hh']=d.ts.dt.floor('h')
for key,win in [('device_id','hh'),('sender_id','hh'),('device_id','hr10'),('sender_id','hr10')]:
    c=d.groupby(['grp',key,win]).size().rename('n').reset_index()
    x=d.merge(c,on=['grp',key,win]); x=x[x.n>=3]
    print('==',key,win,'rows in groups>=3')
    print(x.groupby(['grp',x.cat.where(x.receiver_type=='MERCHANT','P2P:'+x.txn_type)]).agg(rows=('lab','size'),fr=('lab',lambda s:(s!='legit').mean()),vel=('lab',lambda s:(s=='velocity_bot').mean())).unstack(0).round(2))
