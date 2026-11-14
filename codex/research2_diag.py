import os
os.environ['NUMBA_CACHE_DIR']='codex/numba_cache'
import sys,json
sys.path.insert(0,'src')
import numpy as np,pandas as pd
from features import load,LABELS
from sklearn.metrics import roc_auc_score
tr,te,acc,mer=load()
tr['split']='train';te['split']='test'
d=pd.concat([tr,te],ignore_index=True);d['month']=d.timestamp.str[:7]
conn=pd.read_parquet('data/conn.parquet').set_index('txn_id').conn
tr['conn']=tr.txn_id.map(conn).fillna(False)
print('DUPLICATES')
for cols in [['timestamp','sender_id','receiver_id','amount'],['sim_age_hours','device_age_days','sender_account_age_days','balance_before','login_to_txn_sec'],['amount','balance_before']]:
 print(cols,'train',tr.duplicated(cols,False).sum(),'combined',d.duplicated(cols,False).sum())
print('RAW RESIDUAL FEATURES')
for c in ['sim_age_hours','device_age_days','sender_account_age_days','login_to_txn_sec','balance_before','amount']:
 for typ,v in [('integer',np.isclose(tr[c]%1,0)),('cents',np.round(tr[c]*100)%100)]:
  print(c,typ,pd.Series(v).groupby(tr.label).mean().round(3).to_dict())
print('SECONDS BY CLASS',pd.crosstab(tr.label,pd.to_datetime(tr.timestamp).dt.second).to_string())
# Conditional thinning: stable constructed phone ids belong to generator-owned pools; examine per-device monthly burst occupancy.
d['own']=d.device_id.str[2:].str.lstrip('0')==d.sender_id.str[2:].str.lstrip('0')
print('NONOWN PHONE OCCUPANCY')
for mo,g in d[~d.own].groupby('month'):
 sizes=g.groupby('device_id').size()
 print(mo,'rows',len(g),'devices',len(sizes),'hist',sizes.value_counts().sort_index().to_dict())
# Full vs sampled September same phone burst sizes, with time separation <= 10min.
print('PHONE BURSTS')
for nm,g in [('Jul',tr[tr.timestamp.str.startswith('2026-07')]),('Aug',tr[tr.timestamp.str.startswith('2026-08')]),('Sep',tr[tr.timestamp.str.startswith('2026-09')]),('Sep51',tr[tr.timestamp.str.startswith('2026-09')].sample(frac=.51,random_state=0)),('Oct',te[te.timestamp.str.startswith('2026-10')])]:
 g=g.sort_values(['device_id','timestamp']).copy();t=pd.to_datetime(g.timestamp)
 br=(g.device_id!=g.device_id.shift())|((t-t.shift()).dt.total_seconds()>600)
 g['event']=br.cumsum();ag=g.groupby('event').agg(n=('txn_id','size'),senders=('sender_id','nunique'),mean_amt=('amount','mean'))
 ag=ag[ag.senders>=2];print(nm,ag.n.value_counts().sort_index().to_dict(),'n',ag.n.sum(),'events',len(ag),'mean',ag.n.mean())
 ag.to_csv('codex/research2_bursts_'+nm+'.csv',index=False)
print('ROW ORDER CLASS STRUCTURE')
for k in [10,100,1000]:
 b=np.arange(len(tr))//k
 rate=(tr.label!='legit').groupby(b).mean()
 print(k,'observed var',rate.var(),'binomial var',.0366*(1-.0366)/k)
