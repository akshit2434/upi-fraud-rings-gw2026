# estimate per-account hidden opening fraction u; compare accounts with LP / GEN / only-legit rows (senders), and receivers
from load8 import *
df,r=base()
a=pd.read_parquet(C+'aad8.parquet'); df=df.merge(a,on='txn_id')
f=(df.ts-df.ts.dt.normalize()).dt.total_seconds()/86400
aad=df.aad.values
df['lo']=np.where(aad==-1,f+0.5,np.where(aad==0,np.maximum(f-0.5,0),0.0))
df['hi']=np.where(aad==-1,1.0,np.where(aad==0,np.minimum(f+0.5,1),f-0.5))
g=df.groupby('sender_id')
U=pd.DataFrame({'lo':g.lo.max(),'hi':g.hi.min(),'n':g.lo.size(),'nLP':g.g.apply(lambda s:(s=='LP').sum()),'nGEN':g.g.apply(lambda s:(s=='GEN').sum())})
U['w']=U.hi-U.lo; U['u']=(U.lo+U.hi)/2
print(U.w.describe())
x=U[U.w<0.1]
print('accounts with tight u:',len(x))
x['cat']=np.where(x.nGEN>0,'GEN',np.where(x.nLP>0,'LP','legit'))
print(x.groupby('cat').u.describe().round(3))
print(pd.crosstab(pd.cut(x.u,10),x.cat,normalize='columns').round(3))
U.to_parquet(C+'u8.parquet')
