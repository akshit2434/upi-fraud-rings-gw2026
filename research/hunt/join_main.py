import sys; sys.path.insert(0,'/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/src')
from load import *
from features import LABELS
P=np.load('/tmp/P_v4base.npy'); ids=np.load('/tmp/P_v4base_ids.npy',allow_pickle=True)
M=pd.DataFrame(P,columns=LABELS); M['txn_id']=ids
M['pred']=np.array(LABELS)[P.argmax(1)]; M['pfraud']=1-M['legit']
g=pd.read_parquet('gbm_iso_sep.parquet').merge(M[['txn_id','pred','pfraud']],on='txn_id')
print('joined',len(g))
g=g.sort_values('p_iso',ascending=False)
for k in [100,300,1000]:
    t=g.head(k); print(k,'y',t.y.mean().round(3),'main pred fraud',(t.pred!='legit').mean().round(3),'main pfraud med',t.pfraud.median().round(3),'label fraud',(t.label!='legit').mean().round(3))
print(pd.crosstab(g.head(300).label,g.head(300).pred))
