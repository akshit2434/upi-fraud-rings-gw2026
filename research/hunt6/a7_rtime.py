# receiver temporal profile: is LP placed where the receiver is otherwise quiet?
from load import *
df=load()
d=df[['txn_id','receiver_id','receiver_type','t','is_test','label','grp','block']].copy()
d=d.sort_values(['receiver_id','t']).reset_index(drop=True)
tt=d.t.values; rid=d.receiver_id.values
start=np.r_[0,np.flatnonzero(rid[1:]!=rid[:-1])+1]; end=np.r_[start[1:],len(d)]
grp_id=np.repeat(np.arange(len(start)),end-start)
F={}
for w,name in [(3600,'1h'),(6*3600,'6h'),(86400,'1d'),(3*86400,'3d'),(7*86400,'7d'),(30*86400,'30d')]:
    # count of receiver rows within +-w (excluding self) via searchsorted on key*BIG + t
    key=grp_id.astype(np.float64)*1e9+tt
    lo=np.searchsorted(key,key-w,side='left'); hi=np.searchsorted(key,key+w,side='right')
    F['rc_'+name]=hi-lo-1
n=np.repeat(end-start,end-start)-1
F=pd.DataFrame(F); F['rn']=n
span=np.repeat(tt[end-1]-tt[start],end-start)
for c in ['1d','7d','30d']: F['rr_'+c]=F['rc_'+c]/F.rn.replace(0,np.nan)
F['txn_id']=d.txn_id.values
F.to_parquet(C+'rtime.parquet')
m=sepval(df); s=d.merge(m,on='txn_id').merge(F,on='txn_id')
lp=(s.grp=='stealth')&(s.pf<0.05); ll=(s.label=='legit')&(s.pf<0.05)
for rt in ['MERCHANT','P2P']:
    mm=(lp|ll)&(s.receiver_type==rt); x=s[mm]; y=lp[mm].values
    print(rt, 'n LP',y.sum(), {c:round(fauc(y,x[c].astype(float)),3) for c in F.columns if c!='txn_id'})
