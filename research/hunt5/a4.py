import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
df=pd.read_parquet('/tmp/hunt5/rawj.parquet')
df['pair_cnt']=df.groupby(['sender_id','receiver_id']).txn_id.transform('size')
df['rev_cnt']=df.set_index(['receiver_id','sender_id']).index.map(df.groupby(['sender_id','receiver_id']).size()).fillna(0).values
tr=df[df.is_test==0]; tr=tr.assign(grp=np.where(tr.label=='legit','legit',np.where(tr.conn,'C','I')))
for rt in ['MERCHANT','P2P']:
    X=tr[tr.receiver_type==rt]
    print(rt); print(X.groupby('grp').pair_cnt.describe(percentiles=[.25,.5,.75,.9]).round(2))
    print(X.groupby('grp').apply(lambda x:(x.pair_cnt>1).mean()).round(3))
    print(X.groupby('label').apply(lambda x:pd.Series({'pair>1':(x.pair_cnt>1).mean(),'rev>0':(x.rev_cnt>0).mean(),'n':len(x)})).round(3))
    print(X.groupby('grp').rec_cnt.describe(percentiles=[.1,.25,.5,.75]).round(0))
te=df[df.is_test==1]
print('test pair>1', te.groupby('receiver_type').apply(lambda x:(x.pair_cnt>1).mean()))
