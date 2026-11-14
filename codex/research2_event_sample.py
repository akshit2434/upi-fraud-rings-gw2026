import os,sys,time,json
os.environ['NUMBA_CACHE_DIR']='codex/numba_cache'
sys.path.insert(0,'src')
import pandas as pd,numpy as np,lightgbm as lgb
from features import load,build_features,load_cached,apply_drop,LABELS,META,T0
from graph_feats import consecutive_links,hop_next,_union_all
from common import clean_bad,get_conn,apply_relabel,macro_f1,xfit_f1
start=time.time();path='codex/research2_event_frame.parquet'
if not os.path.exists(path):
 tr,te,acc,mer=load();t=(pd.to_datetime(tr.timestamp)-T0).dt.total_seconds().values.astype(np.int64)
 A=[];B=[]
 for k,g in [('device_id',1800),('sender_id',3600),('receiver_id',3600)]:
  a,b=consecutive_links(pd.factorize(tr[k])[0],t,g,valid=(tr.receiver_type.values=='P2P') if k=='receiver_id' else None);A.append(a);B.append(b)
 nxt=hop_next(tr,t);ok=nxt>=0;A.append(np.flatnonzero(ok));B.append(nxt[ok])
 par=np.arange(len(tr),dtype=np.int64);_union_all(par,np.concatenate(A),np.concatenate(B))
 keep=np.random.default_rng(0).random(len(tr))[par]<.51
 df=pd.concat([tr[keep].assign(_is_test=0),te.assign(_is_test=1)],ignore_index=True)
 print('FRAME',len(df),'train',keep.sum(),'component size max',np.bincount(par).max(),flush=True)
 F=build_features(df,acc,mer)
 for k,v in [('_label',df.label),('_month',df.timestamp.str[:7]),('_txn_id',df.txn_id),('_is_test',df._is_test)]:F[k]=v.values
 F.to_parquet(path);del tr,te,df,F,acc,mer
 print('BUILD seconds',time.time()-start,flush=True)
Z=pd.read_parquet(path);zm=Z[META].copy();Z=apply_drop(Z.drop(columns=META))
v=((zm._is_test==0)&(zm._month=='2026-09')).values
yval=zm._label[v].map({l:i for i,l in enumerate(LABELS)}).values
np.save('codex/research2_event_y.npy',yval);np.save('codex/research2_event_ids.npy',zm._txn_id.values[v])
for mode in ['row_sample','event_sample']:
 if mode=='row_sample':X,m=load_cached(0);X=apply_drop(X)
 else:X,m=Z,zm
 ids=m._txn_id.values;y=m._label.map({l:i for i,l in enumerate(LABELS)}).fillna(-1).astype(int).values
 trn=((m._is_test==0)&m._month.isin(['2026-07','2026-08'])).values
 bad=clean_bad(y,get_conn(ids,until='2026-09-01'),'iso7');y=apply_relabel(y,ids,'data/relabel_R1.parquet',trn)
 mask=trn&~bad&((y!=0)|(np.random.default_rng(0).random(len(y))<.25))
 model=lgb.train(dict(objective='multiclass',num_class=12,learning_rate=.05,num_leaves=63,min_child_samples=40,feature_fraction=.7,bagging_fraction=.8,bagging_freq=1,lambda_l2=1.,verbose=-1,num_threads=10,seed=0),lgb.Dataset(X[mask],y[mask],weight=np.where(y[mask]==0,4.,1.)),400)
 P=model.predict(Z[v]);np.save('codex/research2_'+mode+'_eventval_P.npy',P)
 res={'mode':mode,'validation':'same event-sampled September','train_targets':int(mask.sum()),'val_targets':int(v.sum()),'argmax':macro_f1(yval,P.argmax(1)),'xfit':[xfit_f1(P,yval,seed=s) for s in [123,124,125]],'seconds':time.time()-start}
 print(json.dumps(res),flush=True)
 with open('codex/research2_results.jsonl','a') as f:f.write(json.dumps(res)+'\n')
 del model
 if mode=='row_sample':del X,m
