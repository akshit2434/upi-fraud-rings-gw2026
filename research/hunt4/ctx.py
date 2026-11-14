# context stats on: Sep full, Sep uniform-thinned (U), Sep legit-only-thinned (L), Oct test
import pandas as pd, numpy as np, sys
sys.path.insert(0,'/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/research/hunt')
from load import load, STRUCT, fauc
df=load()
def win_count(keys, t, w, side='both'):
    # count of other rows with same key within window w seconds (side: both/past/future)
    o=np.lexsort((t,keys)); k=keys[o]; tt=t[o]
    # composite: offset keys by large to use single searchsorted
    kk=pd.factorize(k)[0].astype(np.float64)*1e9+tt
    lo=kk-(w if side!='future' else 0); hi=kk+(w if side!='past' else 0)
    c=np.searchsorted(kk,hi,'right')-np.searchsorted(kk,lo,'left')-1
    out=np.empty_like(c); out[o]=c; return out
def ctx(d):
    d=d.copy(); t=d.t.values
    S=d.sender_id.values; R=d.receiver_id.values; D=d.device_id.values
    F={}
    for nm,key in [('s',S),('r',R),('d',D),('pair',(d.sender_id+'>'+d.receiver_id).values)]:
        for w,wn in [(3600,'1h'),(86400,'1d'),(7*86400,'7d'),(40*86400,'all')]:
            F[f'{nm}_{wn}']=win_count(key,t,w)
    # receiver acts as sender (onward) / sender as receiver
    allk=np.concatenate([S,R]); allt=np.concatenate([t,t]); role=np.r_[np.zeros(len(S)),np.ones(len(R))]
    for w,wn in [(3600,'1h'),(86400,'1d')]:
        # rows where receiver R sends within w after
        sendt=pd.DataFrame({'k':S,'t':t}).sort_values(['k','t'])
        g=sendt.groupby('k').t.apply(np.array).to_dict()
        def cnt(keys,ts,lo,hi):
            r=np.zeros(len(keys),int)
            for i,(k,x) in enumerate(zip(keys,ts)):
                a=g.get(k)
                if a is not None: r[i]=np.searchsorted(a,x+hi,'right')-np.searchsorted(a,x+lo,'left')
            return r
        F[f'onward_{wn}']=cnt(R,t,0,w)
        recvt=pd.DataFrame({'k':R,'t':t}); g=recvt.groupby('k').t.apply(lambda s:np.sort(s.values)).to_dict()
        F[f'inbefore_{wn}']=cnt(S,t,-w,0)
    F['s_samt']=d.groupby(['sender_id','amount']).t.transform('size').values-1
    F['r_samt']=d.groupby(['receiver_id','amount']).t.transform('size').values-1
    F['s_nrecv']=d.groupby('sender_id').receiver_id.transform('nunique').values
    F['r_nsend']=d.groupby('receiver_id').sender_id.transform('nunique').values
    F['d_nsend']=d.groupby('device_id').sender_id.transform('nunique').values
    F=pd.DataFrame(F,index=d.index); return F
sep=df[(df.is_test==0)&(df.ts>='2026-09-01')&(df.ts<'2026-10-01')]
rng=np.random.default_rng(0)
u=rng.random(len(sep))<0.51
fr=(sep.label!='legit').values
sets={'full':sep,'U':sep[u],'L':sep[u|fr],'test':df[df.is_test==1]}
res={}
for nm,d in sets.items():
    F=ctx(d); F['grp']=d.grp.values; F['lab']=d.label.values; F['struct']=d.label.isin(STRUCT).values
    F.to_parquet(f'/tmp/hunt4/ctx_{nm}.parquet'); print(nm,len(d),flush=True)
