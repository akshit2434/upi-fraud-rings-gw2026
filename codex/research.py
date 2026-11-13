import os
os.environ['NUMBA_CACHE_DIR']='codex/numba_cache'
os.environ['OMP_NUM_THREADS']='6'
import sys,json,time
sys.path.insert(0,'src')
import numpy as np,pandas as pd,lightgbm as lgb
from features import load,sample_frame,Index,T0,LABELS
from common import macro_f1,tune_weights
from sklearn.metrics import f1_score

def score(name,P,y):
    np.save('codex/'+name+'_P.npy',P)
    w,f=tune_weights(P,y,iters=2)
    out=dict(argmax=macro_f1(y,P.argmax(1)),tuned=f,weights=w.tolist(),per_class=f1_score(y,(P*w).argmax(1),average=None,labels=range(12)).tolist())
    print(name,json.dumps(out),flush=True)
    with open('codex/results.jsonl','a') as h:h.write(json.dumps(dict(name=name,**out))+'\n')
    return w

def extra(df):
    t=(pd.to_datetime(df.timestamp)-T0).dt.total_seconds().values.astype('int64')
    ent=pd.Index(pd.unique(np.r_[df.sender_id.values,df.receiver_id.values]));s=ent.get_indexer(df.sender_id);r=ent.get_indexer(df.receiver_id);d=pd.factorize(df.device_id)[0]
    vals={'freshsim':(df.sim_age_hours<24).values.astype(float),'freshdev':(df.device_age_days<3).values.astype(float),'pin':(df.pin_retries>0).values.astype(float),'drain':(df.amount/(df.balance_before+1)>0.8).values.astype(float),'fastlogin':(df.login_to_txn_sec<30).values.astype(float),'notown':(df.device_id.str[2:]!=df.sender_id.str[2:].str.zfill(7)).values.astype(float),'small':(df.amount<=10).values.astype(float),'collect':(df.txn_type=='COLLECT').values.astype(float)}
    inds={k:Index(g,t,vals) for k,g in [('s',s),('r',r),('d',d)]};F={}
    for name,k,q,lo,hi,selfin in [('s1h','s',s,-3600,3600,True),('r1h','r',r,-3600,3600,True),('d1h','d',d,-3600,3600,True),('sin','r',s,-3600,0,False),('rout','s',r,0,3600,False),('s7d','s',s,-604800,604800,True),('r7d','r',r,-604800,604800,True)]:
        idx=inds[k];a,b=idx.span(q,t,lo,hi)
        for vn,v in vals.items():F[name+'_'+vn]=idx.sum(vn,a,b)-(v if selfin else 0)
    return pd.DataFrame(F).astype('float32')

tr,te,acc,mer=load(); df=sample_frame(tr,te,0)
X=pd.read_parquet('/tmp/feat_s0.parquet');assert np.array_equal(df.txn_id,X._txn_id)
y=X._label.map(dict(zip(LABELS,range(12)))).fillna(-1).values.astype(int)
val=((X._is_test==0)&(X._month=='2026-09')).values
train=((X._is_test==0)&X._month.isin(['2026-07','2026-08'])).values
conn=pd.read_parquet('data/conn.parquet').set_index('txn_id').conn.reindex(X._txn_id).fillna(False).values.astype(bool)
bad=(y>0)&~conn&np.isin(y,[1,2,3,4,5,9,11]); rng=np.random.default_rng(0)
mask=train&~bad&((y!=0)|(rng.random(len(X))<.25))
X=X.drop(columns=['_label','_month','_is_test','_txn_id','cat_mismatch','ip_rank_month'],errors='ignore')
np.save('codex/y_val.npy',y[val]);df.loc[val,['txn_id']].to_csv('codex/val_ids.csv',index=False)
E=extra(df);E.to_parquet('codex/extra.parquet');del tr,te,df
params=dict(objective='multiclass',num_class=12,learning_rate=.05,num_leaves=63,min_child_samples=40,feature_fraction=.7,bagging_fraction=.8,bagging_freq=1,lambda_l2=1.,verbose=-1,num_threads=6,seed=0)
for name,XX,mm,yy in [('signature',pd.concat([X,E],axis=1),mask,y),('relabel',X,train&((np.where(bad,0,y)!=0)|(np.random.default_rng(0).random(len(X))<.25)),np.where(bad,0,y))]:
    print('training',name,XX.shape,int(mm.sum()),flush=True)
    m=lgb.train(params,lgb.Dataset(XX[mm],yy[mm],weight=np.where(yy[mm]==0,4.,1.)),400)
    P=m.predict(XX[val],num_threads=6);score(name,P,y[val]);pd.Series(m.feature_importance('gain'),index=XX.columns).sort_values(ascending=False).to_csv('codex/'+name+'_importance.csv')
score('baseline',np.load('/tmp/P_f3_noip_400.npy'),y[val])
