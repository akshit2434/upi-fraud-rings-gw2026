from base import *
o=pd.read_parquet('pu.parquet'); d=load(); s=pd.read_parquet('sepp.parquet')[['txn_id','sp']]
d=d[['txn_id','label','v9','sender_id','receiver_id','device_id','timestamp','amount','txn_type','receiver_type','cat']].merge(s,on='txn_id',how='left')
o=o.merge(d,on='txn_id',how='left'); o['pred']=np.where(o.grp=='oct',o.v9,o.sp)
o['pl']=o.pred!='legit'
S=o[(o.grp=='sep')&(o.mk==0)].sort_values('pu',ascending=False); O=o[(o.grp=='oct')&(o.mk==0)].sort_values('pu',ascending=False)
S['tf']=S.label!='legit'
print('Sep unmarked: top-k by PU score: true fraud rate, among pred-legit')
for k in [100,200,500,1000,2000,5000]:
    t=S.head(k); tl=t[~t.pl]
    print(k, 'fraud',round(t.tf.mean(),3),'pred fraud',round(t.pl.mean(),3),'| pred-legit n',len(tl),'true fraud',round(tl.tf.mean(),3), tl.label[tl.tf].value_counts().head(4).to_dict())
for thr in [0.9,0.7,0.5,0.3,0.2,0.1,0.05]:
    t=S[S.pu>thr]; tl=t[~t.pl]; u=O[O.pu>thr]; ul=u[~u.pl]
    print('thr',thr,'Sep n',len(t),'fraud',round(t.tf.mean(),3),'predlegit n',len(tl),'fr',round(tl.tf.mean(),3),'|| Oct n',len(u),'pred fraud',round(u.pl.mean(),3),'predlegit n',len(ul))
o.to_parquet('pu_j.parquet')
