from base import *
pd.set_option('display.width',250,'display.max_columns',30,'display.max_rows',400)
d=load(); s=pd.read_parquet('sepp.parquet')[['txn_id','sp']]; d=d.merge(s,on='txn_id',how='left')
d['dn']=d.device_id.str[2:].astype(int)
P=np.load('/tmp/P_bag4_R1.npy'); ids=np.load('/tmp/P_bag4_R1_ids.npy',allow_pickle=True); w=np.load(R+'preds/v6_R1_bag_weights.npy')
Q=P*w; fc=np.array(LABELS)[1+Q[:,1:].argmax(1)]
d=d.merge(pd.DataFrame({'txn_id':ids,'fc':fc}),on='txn_id',how='left')
def knn(x,K,pred='pred'):
    x=x.sort_values('dn').reset_index(drop=True); F=x[x[pred]!='legit']; fd=F.dn.values; fl=F[pred].values; fid=F.txn_id.values
    out=[]
    for i,r in x.iterrows():
        dist=np.abs(fd-r.dn).astype(float); dist[fid==r.txn_id]=np.inf
        idx=np.argsort(dist,kind='stable')[:K]
        out.append(pd.Series(fl[idx]).value_counts().index[0])
    x['nb']=out; return x
S=d[(d.grp=='sep')&(d.dn>=420000)&(d.dn<428400)].copy(); S['pred']=S.sp
L0=S[S.pred=='legit']; tf=L0.label!='legit'
print('model best fraud class acc among true fraud', round((L0.fc[tf]==L0.label[tf]).mean(),3))
for K in [1,3,5,9,15]:
    x=knn(S,K); L=x[x.pred=='legit']; tf=L.label!='legit'
    print('K',K,'acc among true fraud',round((L.nb[tf]==L.label[tf]).mean(),3))
x=knn(S,5); L=x[x.pred=='legit']; print(pd.crosstab(L.label,L.nb))
