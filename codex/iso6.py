exec(open('codex/research.py').read().split('tr,te,acc,mer=load();')[0])
from fast_metric import install; install()

X=pd.read_parquet('/tmp/feat_s0.parquet');y=X._label.map(dict(zip(LABELS,range(12)))).fillna(-1).values.astype(int)
val=((X._is_test==0)&(X._month=='2026-09')).values
train=((X._is_test==0)&X._month.isin(['2026-07','2026-08'])).values
conn=pd.read_parquet('data/conn.parquet').set_index('txn_id').conn.reindex(X._txn_id).fillna(False).values.astype(bool)
bad=(y>0)&~conn&np.isin(y,[1,2,3,4,5,11])
mask=train&~bad&((y!=0)|(np.random.default_rng(0).random(len(X))<.25))
X=X.drop(columns=['_label','_month','_is_test','_txn_id','cat_mismatch','ip_rank_month'],errors='ignore')
params=dict(objective='multiclass',num_class=12,learning_rate=.05,num_leaves=63,min_child_samples=40,feature_fraction=.7,bagging_fraction=.8,bagging_freq=1,lambda_l2=1.,verbose=-1,num_threads=6,seed=0)
print('iso6 train rows',mask.sum(),flush=True)
m=lgb.train(params,lgb.Dataset(X[mask],y[mask],weight=np.where(y[mask]==0,4.,1.)),400)
score('iso6',m.predict(X[val],num_threads=6),y[val])
