# hidden account-opening time u: age = round(ts - opened_on - u). Check per-sender consistency of u; LOO conflicts by group
from load8 import *
df,r=base()
a=pd.read_parquet(C+'aad8.parquet'); df=df.merge(a,on='txn_id')
f=(df.ts-df.ts.dt.normalize()).dt.total_seconds()/86400
aad=df.aad.values
lo=np.where(aad==-1,f+0.5,np.where(aad==0,np.maximum(f-0.5,0),0.0))
hi=np.where(aad==-1,1.0,np.where(aad==0,np.minimum(f+0.5,1),f-0.5))
df['lo']=lo; df['hi']=hi
print('row-level empty intervals', (hi<lo).sum())
g=df.groupby('sender_id')
df['n']=g.lo.transform('size')
# LOO max lo / min hi
def top2(s,asc):
    return s
srt=df[['sender_id','lo']].sort_values(['sender_id','lo'],ascending=[True,False])
srt['rk']=srt.groupby('sender_id').cumcount()
m1=srt[srt.rk==0].set_index('sender_id').lo; m2=srt[srt.rk==1].set_index('sender_id').lo
df['maxlo1']=df.sender_id.map(m1); df['maxlo2']=df.sender_id.map(m2).fillna(-1)
srt=df[['sender_id','hi']].sort_values(['sender_id','hi'])
srt['rk']=srt.groupby('sender_id').cumcount()
h1=srt[srt.rk==0].set_index('sender_id').hi; h2=srt[srt.rk==1].set_index('sender_id').hi
df['minhi1']=df.sender_id.map(h1); df['minhi2']=df.sender_id.map(h2).fillna(2)
olo=np.where(df.lo==df.maxlo1,df.maxlo2,df.maxlo1)
ohi=np.where(df.hi==df.minhi1,df.minhi2,df.minhi1)
# conflict: row interval disjoint from intersection of others (only when others consistent)
others_ok=olo<ohi
conf=others_ok&((df.hi<=olo)|(df.lo>=ohi))
df['conf']=conf; df['others_ok']=others_ok
print('senders fully consistent share', (g.lo.max()<g.hi.min()).mean())
m=df.n>=2
print(pd.crosstab(df[m].g,[df[m].others_ok,df[m].conf]).to_string())
print(df[m].groupby('g').conf.mean().round(4))
df[['txn_id','conf','others_ok','n']].to_parquet(C+'uconf8.parquet')
