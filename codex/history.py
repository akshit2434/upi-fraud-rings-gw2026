import sys
sys.path.insert(0,'src')
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from features import LABELS
tr=pd.read_csv('data/upi-fraud-rings/train.csv',usecols=['txn_id','timestamp','sender_id','receiver_id','device_id','label'])
fit=tr[tr.timestamp.str[:7].isin(['2026-07','2026-08'])];val=tr[tr.timestamp.str[:7]=='2026-09'].copy()
conn=pd.read_parquet('data/conn.parquet').set_index('txn_id').conn.reindex(val.txn_id).values.astype(bool)
for lab in LABELS[1:]:
    hist=fit[fit.label==lab]
    scores=[]
    for key in ['sender_id','receiver_id','device_id']:
        count=hist[key].value_counts();total=fit[key].value_counts()
        rate=(count/(total+10)).fillna(0)
        scores.append(val[key].map(rate).fillna(0).values)
    score=np.max(scores,axis=0);sel=(val.label=='legit').values|((val.label==lab).values&~conn)
    yy=(val.label.values[sel]==lab)
    print(lab,'isolated_n',int(yy.sum()),'class_history_auc',roc_auc_score(yy,score[sel]),'isolated_hit',np.mean(score[sel][yy]>0),'legit_hit',np.mean(score[sel][~yy]>0),flush=True)
