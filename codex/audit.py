import os
os.environ['NUMBA_CACHE_DIR']='codex/numba_cache'
import sys,json
sys.path.insert(0,'src')
import numpy as np,pandas as pd
from features import load,sample_frame
tr,te,a,m=load(); x=sample_frame(tr,te,0); C=pd.read_parquet('/tmp/feat_s0.parquet')
print('cache_alignment',len(x),np.array_equal(x.txn_id,C._txn_id),'columns',len(C.columns))
print('duplicate_ids_train',tr.txn_id.duplicated().sum(),'test',te.txn_id.duplicated().sum(),'overlap',len(set(tr.txn_id)&set(te.txn_id)))
print('shared_y_equals_cache',np.array_equal(np.load('/tmp/y_val.npy'),C.loc[(C._is_test==0)&(C._month=='2026-09'),'_label'].map({l:i for i,l in enumerate(__import__('features').LABELS)}).values))
print('month stats: rows, distinct senders, repeat sender share, non-own device share, QR mismatch, ref')
for name,z in [('jul',tr[tr.timestamp.str[:7]=='2026-07']),('aug',tr[tr.timestamp.str[:7]=='2026-08']),('sep',tr[tr.timestamp.str[:7]=='2026-09']),('sep51',x[(x._is_test==0)&(x.timestamp.str[:7]=='2026-09')]),('oct',te[te.timestamp.str[:7]=='2026-10'])]:
    print(name,len(z),z.sender_id.nunique(),z.sender_id.duplicated(False).mean(),(z.device_id.str[2:]!=z.sender_id.str[2:].str.zfill(7)).mean(),(z.qr_vpa_mismatch==1).mean(),(z.txn_type=='REFUND').mean())
print('cat mismatch train by label')
cat=m.set_index('merchant_id').category
for name,z in [('train',tr),('test',te)]:
    z=z[z.receiver_type=='MERCHANT'].copy();z['mismatch']=(z.receiver_id.map(cat)!=z.merchant_category)
    print(name,z.groupby('label').mismatch.agg(['mean','count']).to_string() if name=='train' else z.mismatch.mean())
# Explicit boundary leakage in label-dependent cleaning: Jul/Aug records whose only same-label connected partner is Sep.
f=tr[tr.label!='legit'].copy();f['t']=pd.to_datetime(f.timestamp); f['early']=f.timestamp.str[:7].isin(['2026-07','2026-08']); connected=set();past=set()
for left,right in [('sender_id','sender_id'),('receiver_id','receiver_id'),('device_id','device_id'),('sender_id','receiver_id'),('receiver_id','sender_id')]:
    # Only boundary rows can differ. All direction matches overestimates causal chain case, report upper bound.
    g=f[(f.t>='2026-08-31 18:00')&(f.t<'2026-09-01 06:00')]
    q=g.merge(g,left_on=[left,'label'],right_on=[right,'label'],suffixes=('_a','_b'))
    q=q[(q.txn_id_a!=q.txn_id_b)&((q.t_a-q.t_b).abs()<pd.Timedelta(hours=6))&q.early_a&~q.early_b]
    connected.update(q.txn_id_a)
print('upper_bound_train_rows_with_sep_label_connection',len(connected))
print('isolated fraud assumption: AUC near .55 is evidence of weak current features, NOT proof of generator origin or irreducibility')
