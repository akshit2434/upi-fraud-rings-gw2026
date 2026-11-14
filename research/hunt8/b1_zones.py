# device zones in test vs train; reuse of train devices by new senders
from load8 import *
df,r=base()
v=pd.read_csv(R+'submissions/v13_oofu.csv').rename(columns={'id':'txn_id','label':'pred'})
df=df.merge(v,on='txn_id',how='left')
dn=df.devn
z=np.select([df.own, dn<420000, dn<=428454, dn<473632, dn<=474551, dn<=482013],['own','acct-range-other','trblock','trsecond','octblock','octsecond'],'beyond')
df['zone']=z
print(pd.crosstab(df.zone,df.is_test))
tr=df[df.is_test==0]; te=df[df.is_test==1]
print(pd.crosstab(tr.zone,tr.label=='legit'))
print(pd.crosstab(te.zone,te.pred))
# train devices reused in test: same sender set?
trs=tr.groupby('device_id').sender_id.agg(set)
te2=te[te.device_id.isin(trs.index)&(te.zone!='own')].copy()
te2['seen_pair']=[s in trs[d] for d,s in zip(te2.device_id,te2.sender_id)]
print(pd.crosstab([te2.zone,te2.seen_pair],te2.pred))
# own-range other: device number < 420000 but != sender
x=df[df.zone=='acct-range-other']
print(x.groupby('is_test').size(), x.label.value_counts().head())
print(df[df.zone=='beyond'][['device_id','is_test','label','pred']].head(20))
print('max devn train/test', tr.devn.max(), te.devn.max())
