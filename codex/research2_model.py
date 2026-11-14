import os
os.environ['NUMBA_CACHE_DIR']='codex/numba_cache'
import sys,time,json,hashlib
sys.path.insert(0,'src')
import pandas as pd,numpy as np,lightgbm as lgb
from features import load,load_cached,apply_drop,LABELS
from common import clean_bad,get_conn,apply_relabel,macro_f1,xfit_f1
mode=sys.argv[1] if len(sys.argv)>1 else 'augmented'
t0=time.time();X,meta=load_cached(0);X=apply_drop(X,'ip_rank_month');ids=meta._txn_id.values
tr,te,acc,mer=load();d=pd.concat([tr,te],ignore_index=True).set_index('txn_id').reindex(ids)
a=acc.set_index('account_id');m=mer.set_index('merchant_id');new=[]
def add(k,v):
 X[k]=np.asarray(v,dtype=np.float32);new.append(k)
if mode!='unclean':
 ts=pd.to_datetime(d.timestamp)
 add('raw_second',ts.dt.second)
 for c in ['sim_age_hours','device_age_days','sender_account_age_days','login_to_txn_sec','balance_before']:
  add(c+'_frac',d[c]%1);add(c+'_last2',np.round(d[c]*100)%100)
 for c in ['device_city']:
  add(c+'_code',pd.Categorical(d[c]).codes)
 for c in ['bank','home_city','os','kyc_level']:
  if c in a:
   cats=sorted(a[c].dropna().unique())
   sv=d.sender_id.map(a[c]);rv=d.receiver_id.map(a[c]);add('s_'+c,pd.Categorical(sv,categories=cats).codes);add('r_'+c,pd.Categorical(rv,categories=cats).codes);add('eq_'+c,sv==rv)
 for c in ['amount','balance_before','sim_age_hours','device_age_days','login_to_txn_sec']:
  for g in ['txn_type','receiver_type','merchant_category']:
   add(c+'_rank_'+g,d.groupby([d.timestamp.str[:7],d[g]])[c].rank(pct=True))
 for lim in [500,1000,2000,2500,5000,10000,20000,50000]:
  add('amt_dist_'+str(lim),np.abs(d.amount-lim));add('amt_mod_'+str(lim),d.amount%lim)
lab=meta._label.map({l:i for i,l in enumerate(LABELS)}).fillna(-1).astype(int).values
trn=((meta._is_test==0)&meta._month.isin(['2026-07','2026-08'])).values
val=((meta._is_test==0)&(meta._month=='2026-09')).values
bad=clean_bad(lab,get_conn(ids,until='2026-09-01'),'none' if mode in ['unclean','augmented_unclean'] else 'iso7')
y=lab if mode in ['unclean','augmented_unclean'] else apply_relabel(lab,ids,'data/relabel_R1.parquet',trn)
mask=trn&~bad&((y!=0)|(np.random.default_rng(0).random(len(y))<.25))
del d,tr,te
model=lgb.train(dict(objective='multiclass',num_class=12,learning_rate=.05,num_leaves=63,min_child_samples=40,feature_fraction=.7,bagging_fraction=.8,bagging_freq=1,lambda_l2=1.,verbose=-1,num_threads=10,seed=0),lgb.Dataset(X[mask],y[mask],weight=np.where(y[mask]==0,4.,1.)),400)
P=model.predict(X[val]);np.save('codex/research2_'+mode+'_P.npy',P);np.save('codex/research2_val_ids.npy',ids[val]);np.save('codex/research2_y.npy',lab[val])
res={'mode':mode,'rows':int(mask.sum()),'features':X.shape[1],'argmax':macro_f1(lab[val],P.argmax(1)),'xfit':[xfit_f1(P,lab[val],seed=s) for s in [123,124,125]],'seconds':time.time()-t0}
print(json.dumps(res),flush=True)
pd.DataFrame({'feature':X.columns,'gain':model.feature_importance('gain')}).sort_values('gain',ascending=False).to_csv('codex/research2_'+mode+'_importance.csv',index=False)
with open('codex/research2_results.jsonl','a') as f:f.write(json.dumps(res)+'\n')
