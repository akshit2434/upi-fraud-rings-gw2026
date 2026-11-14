# P2P graph structure: common neighbours of sender & receiver, receiver's degree, reverse edge
from load import *
import scipy.sparse as sp
df=load()
p=df[df.receiver_type=='P2P'][['txn_id','sender_id','receiver_id','label','grp','is_test']].copy()
acc=pd.Index(pd.unique(np.r_[p.sender_id.values,p.receiver_id.values]))
si=acc.get_indexer(p.sender_id); ri=acc.get_indexer(p.receiver_id)
n=len(acc)
A=sp.coo_matrix((np.ones(len(p)),(si,ri)),shape=(n,n)).tocsr()
U=((A+A.T)>0).astype(np.float32).tocsr()
# common neighbours via row-wise dot (U[s] . U[r])
cn=np.asarray(U[si].multiply(U[ri]).sum(1)).ravel()
deg=np.asarray(U.sum(1)).ravel()
rev=np.asarray(A[ri,si]).ravel()>0
pairn=np.asarray(A[si,ri]).ravel()
p['cn']=cn; p['sdeg']=deg[si]; p['rdeg']=deg[ri]; p['rev']=rev; p['pairn']=pairn
# Mutual: do sender and receiver pay a common merchant?
m=df[df.receiver_type=='MERCHANT']
mi=pd.Index(m.receiver_id.unique()); B=sp.coo_matrix((np.ones(len(m)),(acc.get_indexer(m.sender_id).clip(0),mi.get_indexer(m.receiver_id))),shape=(n,len(mi))).tocsr()
B=(B>0).astype(np.float32)
p['cm']=np.asarray(B[si].multiply(B[ri]).sum(1)).ravel()
p[['txn_id','cn','sdeg','rdeg','rev','pairn','cm']].to_parquet(C+'p2p_graph.parquet')
mm=sepval(df); s=p.merge(mm,on='txn_id')
lp=(s.grp=='stealth')&(s.pf<0.05); ll=(s.label=='legit')&(s.pf<0.05)
x=s[lp|ll]; y=lp[lp|ll].values
print('n LP',y.sum())
for c in ['cn','sdeg','rdeg','rev','pairn','cm']:
    print(c, round(fauc(y,x[c].astype(float)),3), x.groupby(y)[c].mean().round(3).to_dict())
print(pd.crosstab(x.cn.clip(0,3),y,normalize='columns'))
