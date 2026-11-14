from base import *
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
f=pd.read_parquet('/tmp/feat_v4_s0.parquet')
f=f[(f._month=='2026-09')|(f._is_test==1)].copy()
f['grp']=np.where(f._is_test==1,'oct','sep')
f=f[f.txn_type!=1.0]
f['mk']=f.qr_vpa_mismatch.notna().astype(int)
drop=['qr_vpa_mismatch','qr_checked','ip_rank_month','cat_mismatch','_label','_month','_is_test','_txn_id','grp','mk']
X=f.drop(columns=drop)
for c in X.columns:
    if X[c].dtype==object or str(X[c].dtype).startswith('str'): X[c]=X[c].astype('category')
out=[]
for g in ['sep','oct']:
    m=(f.grp==g).values; Xg=X[m]; y=f.mk.values[m]; oof=np.zeros(m.sum())
    for k,(a,b) in enumerate(StratifiedKFold(4,shuffle=True,random_state=0).split(Xg,y)):
        mdl=lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=20,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1)
        mdl.fit(Xg.iloc[a],y[a]); oof[b]=mdl.predict_proba(Xg.iloc[b])[:,1]
    out.append(pd.DataFrame({'txn_id':f._txn_id.values[m],'grp':g,'mk':y,'pu':oof}))
    print(g,'done', y.mean())
o=pd.concat(out); o.to_parquet('/tmp/hunt3/pu.parquet')
