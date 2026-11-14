# 2-fold CV GBM: LP (Sep stealth, model-legit) vs LL (model-legit legit) on raw + entity + deviation features
from load import *
import lightgbm as lgb
df=load(); m=sepval(df)
s=df.merge(m,on='txn_id')
for f in ['ent_feats','dev_feats','rtime']:
    s=s.merge(pd.read_parquet(C+f+'.parquet'),on='txn_id',how='left')
lp=(s.grp=='stealth')&(s.pf<0.05); ll=(s.label=='legit')&(s.pf<0.05)
s=s[lp|ll].reset_index(drop=True); y=((s.grp=='stealth')).values.astype(int)
s['hour']=s.ts.dt.hour; s['dow']=s.ts.dt.dayofweek; s['day']=s.ts.dt.day
for c in ['receiver_type','txn_type','merchant_category','device_city','app']: s[c]=s[c].astype('category')
drop=['txn_id','timestamp','sender_id','receiver_id','device_id','label','ts','grp','is_test','conn','block','pf','app_version','ip_risk_score','mon','t']
X=s.drop(columns=[c for c in drop if c in s.columns])
rng=np.random.RandomState(0); fold=rng.randint(0,2,len(s))
p=np.zeros(len(s))
for k in [0,1]:
    tr=fold!=k
    mdl=lgb.train(dict(objective='binary',learning_rate=0.03,num_leaves=15,min_data_in_leaf=200,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,verbose=-1),lgb.Dataset(X[tr],y[tr]),300)
    p[~tr]=mdl.predict(X[~tr])
print('CV AUC',fauc(y,p))
imp=pd.Series(mdl.feature_importance('gain'),X.columns).sort_values(ascending=False); print((imp/imp.sum()).head(15).round(3))
o=np.argsort(-p)
for k in [50,200,1000]: print('prec@',k,y[o[:k]].mean().round(3),'base',y.mean().round(4))
