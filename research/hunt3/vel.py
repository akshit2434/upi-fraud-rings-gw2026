from base import *
d=load(); d=d[d.grp!='aug']
pd.set_option('display.width',250,'display.max_columns',30,'display.max_rows',200)
d['lab']=np.where(d.grp=='oct',d.v9,d.label); d['day']=d.ts.dt.date
x=d[d.cat.isin(['ecommerce','gaming'])&(d.txn_type!='REFUND')].copy()
for key in ['sender_id','device_id','receiver_id']:
    c=x.groupby(['grp',key,'day']).size().rename('n'); x=x.merge(c.reset_index().rename(columns={'n':'n_'+key[:3]}),on=['grp',key,'day'])
for key in ['n_sen','n_dev']:
    print(key)
    t=x.groupby(['grp',x[key].clip(upper=8)]).agg(rows=('lab','size'),vel=('lab',lambda s:(s=='velocity_bot').mean()),fr=('lab',lambda s:(s!='legit').mean())).unstack(0)
    print(t.round(3))
# merchant-day counts from distinct senders, small amounts
print('velocity labelled rows: own device share, n_dev dist'); 
for g in ['sep','oct']:
    v=x[(x.grp==g)&(x.lab=='velocity_bot')]; print(g,len(v),'own',round(v.device_is_own.mean(),2),'n_dev med',v.n_dev.median(),'n_sen med',v.n_sen.median(),'login med',v.login_to_txn_sec.median(),'amt med',v.amount.median())
x.to_parquet('vel.parquet')
