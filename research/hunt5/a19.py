import pandas as pd, numpy as np
P='/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/'
df=pd.read_parquet('/tmp/hunt5/raw.parquet')
v10=pd.read_csv(P+'submissions/v10_v6_octrules.csv').rename(columns={'id':'txn_id','label':'pred'})
df=df.merge(v10,on='txn_id',how='left'); df['dn']=df.device_id.str[2:].astype(int)
def nprob(x):  # distribution of rows/device by class, train block
    return x
tr=df[(df.is_test==0)&(df.dn>=420000)&(df.dn<428455)]
d=tr[tr.label!='legit'].groupby('dn').label.agg(lambda s:s.value_counts().index[0]); n=tr.groupby('dn').size()
D=pd.DataFrame({'lab':d,'n':n}).dropna()
pS=D[D.lab=='sim_swap'].n.clip(upper=4).value_counts(normalize=True).sort_index(); pA=D[D.lab=='account_takeover'].n.clip(upper=4).value_counts(normalize=True).sort_index()
print(pS.round(3).to_dict(), pA.round(3).to_dict())
te=df[(df.is_test==1)&(df.dn>=473916)&(df.dn<474283)]
O=te.groupby('dn').agg(n=('txn_id','size'),labs=('pred',lambda s:'/'.join(sorted(set(s)))))
O=O[O.labs.str.contains('account_takeover|sim_swap')]
print(len(O),'devices', O.n.sum(),'rows')
ns=O.n.clip(upper=4).values
ll=[np.log(pS.reindex(ns[:k]).values).sum()+np.log(pA.reindex(ns[k:]).values).sum() for k in range(len(ns)+1)]
ll=np.array(ll); k=ll.argmax(); print('changepoint k',k,'device',O.index[k] if k<len(O) else None,'ll gain vs all-ATO',ll[k]-ll[0])
print('top ks', np.argsort(-ll)[:10], [O.index[i] for i in np.argsort(-ll)[:10]])
k38=int(round(0.38*len(O))); print('38% device quantile k',k38,'device',O.index[k38])
# profile-only test for sim rows wrongly predicted in Sep? -> how v6 typed train-block sim rows by n
print(O.head(60).to_string())
