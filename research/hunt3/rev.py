from base import *
d=load(); d=d[d.grp!='aug']
pd.set_option('display.width',250,'display.max_columns',30,'display.max_rows',200)
d['t']=(d.ts-pd.Timestamp('2026-07-01')).dt.total_seconds()
res=[]
for g in ['sep','oct']:
    x=d[d.grp==g][['txn_id','sender_id','receiver_id','t','amount','txn_type','label','v9','receiver_type']].copy()
    p=x[x.receiver_type=='P2P']
    # reverse: earlier payment from receiver to sender
    r=p[['sender_id','receiver_id','t','amount','txn_type']].rename(columns={'sender_id':'receiver_id','receiver_id':'sender_id','t':'rt','amount':'ramt','txn_type':'rtype'})
    m=p.merge(r,on=['sender_id','receiver_id'])
    m=m[(m.rt<m.t)&(m.t-m.rt<7*86400)]
    m=m.sort_values('rt').groupby('txn_id').tail(1)  # most recent reverse
    m['grp']=g; res.append(m)
m=pd.concat(res)
m['lab']=np.where(m.grp=='oct',m.v9,m.label)
m['rb']=pd.cut(m.ramt,[0,10,50,100,500,2000,1e9]).astype(str); m['dtb']=pd.cut((m.t-m.rt)/3600,[0,1,6,24,72,168]).astype(str)
m['big']=m.amount>m.ramt*5
for g in ['sep','oct']:
    y=m[m.grp==g]; print(g, len(y))
    print(y.groupby(['rb','big']).agg(n=('lab','size'),cash=('lab',lambda s:(s=='fake_cashback').mean()),fraud=('lab',lambda s:(s!='legit').mean())).round(2))
y=m[(m.ramt<=10)]
print(y.groupby(['grp','dtb']).agg(n=('lab','size'),cash=('lab',lambda s:(s=='fake_cashback').mean()),fraud=('lab',lambda s:(s!='legit').mean())).round(2))
print(m[(m.grp=='oct')&(m.ramt<=10)].lab.value_counts(), m[(m.grp=='sep')&(m.ramt<=10)].lab.value_counts())
m.to_parquet('rev.parquet')
