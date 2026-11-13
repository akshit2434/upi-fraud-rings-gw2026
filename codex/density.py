# Full-density training context, same training targets and held-out September targets.
exec(open('codex/research.py').read().split('tr,te,acc,mer=load();')[0])
from features import build_features
from fast_metric import install; install()

tr,te,acc,mer=load()
base=pd.read_parquet('/tmp/feat_s0.parquet'); meta=base[['_txn_id','_label','_month','_is_test']]
cols=base.drop(columns=['_txn_id','_label','_month','_is_test','cat_mismatch','ip_rank_month'],errors='ignore').columns
v=((meta._is_test==0)&(meta._month=='2026-09')).values
y=meta._label.map(dict(zip(LABELS,range(12)))).fillna(-1).values.astype(int)
conn=pd.read_parquet('data/conn.parquet').set_index('txn_id').conn.reindex(meta._txn_id).fillna(False).values.astype(bool)
bad=(y>0)&~conn&np.isin(y,[1,2,3,4,5,9,11]);train=((meta._is_test==0)&meta._month.isin(['2026-07','2026-08'])).values
mask=train&~bad&((y!=0)|(np.random.default_rng(0).random(len(base))<.25))
full=pd.concat([tr.assign(_is_test=0),te.assign(_is_test=1)],ignore_index=True)
print('build full density',len(full),flush=True)
X=build_features(full,acc,mer);X.index=full.txn_id
X=X.reindex(meta._txn_id)[cols].reset_index(drop=True)
X.to_parquet('codex/full_density_same_targets.parquet');del full,tr,te
params=dict(objective='multiclass',num_class=12,learning_rate=.05,num_leaves=63,min_child_samples=40,feature_fraction=.7,bagging_fraction=.8,bagging_freq=1,lambda_l2=1.,verbose=-1,num_threads=6,seed=0)
print('train full density',flush=True)
m=lgb.train(params,lgb.Dataset(X[mask],y[mask],weight=np.where(y[mask]==0,4.,1.)),400)
score('fulltrain_sampledval',m.predict(base.loc[v,cols],num_threads=6),y[v])
score('fulltrain_fullval_optimistic',m.predict(X[v],num_threads=6),y[v])
