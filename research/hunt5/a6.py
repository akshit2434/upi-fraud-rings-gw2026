import pandas as pd, numpy as np
df=pd.read_parquet('/tmp/hunt5/raw.parquet')
df=df.sort_values(['sender_id','t']); g=df.groupby('sender_id')
df['pb']=g.balance_before.shift(1); df['pa']=g.amount.shift(1); df['nb']=g.balance_before.shift(-1)
df['r1']=df.balance_before/df.pb
print(df[['sender_id','t','balance_before','amount','label']].head(20).to_string())
tr=df[df.is_test==0]; tr=tr.assign(grp=np.where(tr.label=='legit','legit',np.where(tr.conn,'C','I')))
print(tr.groupby('grp').r1.describe(percentiles=[.05,.25,.5,.75,.95]))
# sender-level balance dispersion
df['lb']=np.log1p(df.balance_before)
df['lb_mean']=g.lb.transform('mean'); df['lb_std']=g.lb.transform('std')
df['z']=(df.lb-(df.lb_mean*g.lb.transform('size')-df.lb)/(g.lb.transform('size')-1))
tr=df[df.is_test==0]; tr=tr.assign(grp=np.where(tr.label=='legit','legit',np.where(tr.conn,'C','I')))
print(tr.groupby('grp').z.describe(percentiles=[.05,.25,.5,.75,.95]))
