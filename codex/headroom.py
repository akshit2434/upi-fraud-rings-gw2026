import sys,json
sys.path.insert(0,'src')
import numpy as np,pandas as pd
from features import LABELS
from common import macro_f1
C=pd.read_parquet('/tmp/feat_s0.parquet');v=((C._is_test==0)&(C._month=='2026-09')).values
y=C._label[v].map(dict(zip(LABELS,range(12)))).values.astype(int)
P=np.load('/tmp/P_f3_noip_400.npy'); results=[json.loads(x) for x in open('codex/results.jsonl')];w=[x['weights'] for x in results if x['name']=='baseline'][-1];p=(P*w).argmax(1)
full=pd.read_parquet('data/conn.parquet').set_index('txn_id').conn.reindex(C._txn_id[v]).values.astype(bool)
f=pd.read_csv('data/upi-fraud-rings/train.csv',usecols=['txn_id','timestamp','label','sender_id','receiver_id','device_id']);f=f[(f.label!='legit')&f.txn_id.isin(C._txn_id)].copy();f['t']=pd.to_datetime(f.timestamp)
found=set()
for a,b,sg in [('sender_id','sender_id',0),('receiver_id','receiver_id',0),('device_id','device_id',0),('receiver_id','sender_id',1),('sender_id','receiver_id',-1)]:
    z=f.merge(f,left_on=[a,'label'],right_on=[b,'label'],suffixes=('_a','_b'));dt=(z.t_b-z.t_a).dt.total_seconds()
    ok=(dt.abs()<21600) if sg==0 else (sg*dt).between(0,21600)
    found.update(z.loc[ok&(z.txn_id_a!=z.txn_id_b),'txn_id_a'])
sampled=C._txn_id[v].isin(found).values
print('base',macro_f1(y,p))
for name,c in [('full_context_connected',full),('sampled_context_connected',sampled)]:
    oracle=p.copy();oracle[c]=y[c]
    print(name,'fraud rows',int(c.sum()),'missed_as_legit',int((c&(p==0)).sum()),'wrong_fraud_type',int((c&(p>0)&(p!=y)).sum()),'oracle_f1',macro_f1(y,oracle))
    print(pd.DataFrame({'label':np.array(LABELS)[y],'c':c,'correct':p==y}).query('c').groupby('label').correct.agg(['count','mean']).to_string())
