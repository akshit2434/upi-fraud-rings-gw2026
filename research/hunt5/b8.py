import pandas as pd, numpy as np
df=pd.read_parquet('/tmp/hunt5/raw.parquet'); tr=df[df.is_test==0].copy()
tr['grp']=np.where(tr.label=='legit','legit',np.where(tr.conn,'C','I'))
C=np.sort(tr[tr.grp=='C'].t.values)
def mind(x):
    i=np.searchsorted(C,x); a=np.abs(C[np.clip(i,0,len(C)-1)]-x); b=np.abs(C[np.clip(i-1,0,len(C)-1)]-x); return np.minimum(a,b)
for g in ['I','legit']:
    x=tr[tr.grp==g].t.values
    if g=='legit': x=np.random.default_rng(0).choice(x,50000,replace=False)
    d=mind(x); print(g, {k:round((d<=k).mean(),4) for k in [0,1,10,60,600]})
# opened_on monotone in account id?
acc=pd.read_csv('/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/data/upi-fraud-rings/accounts.csv')
acc['o']=pd.to_datetime(acc.opened_on); print('opened_on vs id spearman', acc.o.rank().corr(pd.Series(range(len(acc))).rank()))
