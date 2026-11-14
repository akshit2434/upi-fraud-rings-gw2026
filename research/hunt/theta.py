from load import *
df=load()
acc=pd.read_csv(R+'data/upi-fraud-rings/accounts.csv')
df=df.merge(acc[['account_id','opened_on']],left_on='sender_id',right_on='account_id',how='left')
d=df.ts.dt.normalize(); op=pd.to_datetime(df.opened_on)
df['diff']=df.sender_account_age_days-(d-op).dt.days
df['f']=(df.ts-d).dt.total_seconds()/86400
print(df['diff'].value_counts())
# interval for theta (theta in [0,1))
lo=np.where(df['diff']==-1,df.f+0.5,np.where(df['diff']==0,df.f-0.5,0.0))
hi=np.where(df['diff']==-1,1.0,np.where(df['diff']==0,df.f+0.5,df.f-0.5))
df['lo']=np.clip(lo,0,1); df['hi']=np.clip(hi,0,1)
g=df.groupby('sender_id')
# leave-one-out intersection using top-2 of lo and bottom-2 of hi
def loo_max(s,grp):
    mx=grp[s.name].transform('max')
    return mx
df=df.sort_values('sender_id')
G=df.groupby('sender_id')
mx1=G.lo.transform('max'); mn1=G.hi.transform('min')
# second max/min
r_lo=G.lo.rank(method='first',ascending=False); r_hi=G.hi.rank(method='first')
mx2=df.lo.where(r_lo==2).groupby(df.sender_id).transform('max'); mn2=df.hi.where(r_hi==2).groupby(df.sender_id).transform('min')
loo_lo=np.where(r_lo==1,mx2,mx1); loo_hi=np.where(r_hi==1,mn2,mn1)
n=G.lo.transform('size')
df['incons']=np.where(n>1,((df.lo>loo_hi+1e-9)|(df.hi<loo_lo-1e-9)).astype(float),np.nan)
df['others_empty']=np.where(n>1,(loo_lo>loo_hi+1e-9).astype(float),np.nan)
print('senders w/ empty full intersection:', (mx1>mn1+1e-9).groupby(df.sender_id).first().mean())
df['k']=np.where(df.grp=='legit','legit',np.where(df.grp=='test','test',df.label+'|'+df.grp))
print(df.groupby('k')[['incons','others_empty']].mean().round(4).to_string())
df[['txn_id','diff','f','incons','others_empty']].to_parquet('/tmp/hunt/theta.parquet')
