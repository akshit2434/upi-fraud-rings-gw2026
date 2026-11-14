# inside train fraud block: what do legit-looking fraud rows keep from their event?
from load8 import *
df,r=base()
tr=df[(df.is_test==0)&(df.devn>=420000)&(df.devn<=428454)].copy()
print(tr.g.value_counts())
bad=set(pd.read_parquet(R+'data/oofbadfull_u.parquet').txn_id)
tr['lpb']=tr.txn_id.isin(bad)&(tr.label!='legit')
print('block fraud rows with OOF P(legit) high:', tr.lpb.sum(), 'of', (tr.label!='legit').sum())
tr=tr.sort_values(['device_id','t'])
g=tr.groupby('device_id')
tr['ndev']=g.t.transform('size'); tr['nsend']=g.sender_id.transform('nunique')
tr['dt_prev']=tr.t-g.t.shift(1); tr['dt_next']=g.t.shift(-1)-tr.t
tr['own']=tr.own
pd.set_option('display.width',250)
cols=['receiver_type','txn_type','own']
for k,sub in [('LPblock',tr[tr.lpb]),('GENblock',tr[(~tr.lpb)&(tr.label!='legit')]),('legit',tr[tr.label=='legit'])]:
    print('==',k,len(sub))
    print(' rt',sub.receiver_type.value_counts(normalize=True).round(3).to_dict(),' tt',sub.txn_type.value_counts(normalize=True).round(3).to_dict())
    print(' ndev med',sub.ndev.median(),' nsend med',sub.nsend.median(),' min gap h',(np.fmin(sub.dt_prev,sub.dt_next)/3600).describe(percentiles=[.1,.5,.9]).round(2).to_dict())
    print(' sim',sub.sim_age_hours.median(),' devage',sub.device_age_days.median(),' login',sub.login_to_txn_sec.median(),' amt',sub.amount.median())
x=tr[tr.device_id.isin(tr[tr.lpb].device_id.unique()[:15])]
print(x[['device_id','timestamp','sender_id','receiver_id','txn_type','amount','balance_before','sim_age_hours','device_age_days','login_to_txn_sec','label','lpb']].to_string())
