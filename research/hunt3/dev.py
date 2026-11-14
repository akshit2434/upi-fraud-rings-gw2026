from base import *
tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv'); te=pd.read_csv(R+'data/upi-fraud-rings/test.csv')
v9=pd.read_csv(R+'submissions/v9_v6_qrrule.csv').rename(columns={'id':'txn_id','label':'v9'}); te=te.merge(v9,on='txn_id'); te['label']=te.v9
pd.set_option('display.width',250,'display.max_columns',30,'display.max_rows',300)
for x in (tr,te):
    x['dn']=x.device_id.str[2:].astype(int); x['own']=x.device_id.str[3:]==x.sender_id.str[2:]; x['mon']=x.timestamp.str[:7]
first=pd.concat([tr,te]).groupby('device_id').timestamp.min().str[:7].rename('dfirst')
for x in (tr,te): x['dfirst']=x.device_id.map(first)
a=pd.concat([tr.assign(src='tr'),te.assign(src='te')])
a=a[~a.own]
a['newdev']=a.dfirst==a.mon
a['blk']=pd.cut(a.dn,[0,420000,428500,473631,1e7],labels=['<420k','fraudblk','428.5-473.6k','>473.6k']).astype(str)
print(a.groupby(['mon','blk']).agg(n=('label','size'),fr=('label',lambda s:(s!='legit').mean())).unstack(1).round(3))
print(a.groupby(['mon','newdev']).agg(n=('label','size'),fr=('label',lambda s:(s!='legit').mean())).unstack(1).round(3))
# rows per non-own device in Oct by block
z=a[a.mon=='2026-10']; print(z.groupby('blk').device_id.nunique()); print(z.groupby('blk').agg(dev_age=('device_age_days','median'),simage=('sim_age_hours','median'),accts=('sender_id','nunique')))
print(pd.crosstab(z.blk,z.label))
for m in ['2026-08','2026-09']:
    y=a[a.mon==m]; print(m); print(pd.crosstab(y.blk,y.label))
