import sys
sys.path.insert(0,'src')
import pandas as pd,numpy as np
from features import load,LABELS
tr,te,_,_=load()
sub=pd.read_csv('submissions/v6_R1_bag.csv').set_index('id').label
records=[]
for nm,g0 in [('Jul',tr[tr.timestamp.str.startswith('2026-07')]),('Aug',tr[tr.timestamp.str.startswith('2026-08')]),('Sep',tr[tr.timestamp.str.startswith('2026-09')]),('Sep51',tr[tr.timestamp.str.startswith('2026-09')].sample(frac=.51,random_state=0)),('Oct',te[te.timestamp.str.startswith('2026-10')])]:
 for kind,key,window in [('phone','device_id',600),('p2p_fanin','receiver_id',3600),('sender','sender_id',1800)]:
  g=g0[g0.receiver_type=='P2P'].copy() if kind=='p2p_fanin' else g0.copy()
  g=g.sort_values([key,'timestamp']);t=pd.to_datetime(g.timestamp)
  g['event']=((g[key]!=g[key].shift())|((t-t.shift()).dt.total_seconds()>window)).cumsum()
  ag=g.groupby('event').agg(n=('txn_id','size'),senders=('sender_id','nunique'),receivers=('receiver_id','nunique'),sim=('sim_age_hours','median'),dev=('device_age_days','median'))
  keep=ag.index[(ag.senders>=3) if kind!='sender' else ((ag.n>=2)&((ag.sim<48)|(ag.dev<7)))]
  if kind=='phone':keep=ag.index[(ag.senders>=3)&(ag.n>=10)]
  q=g[g.event.isin(keep)];n=ag.loc[keep].n
  rec={'set':nm,'kind':kind,'events':len(keep),'rows':len(q),'mean_size':n.mean(),'size_hist':n.value_counts().sort_index().to_dict(),'labels':q.label.value_counts().to_dict() if nm!='Oct' else q.txn_id.map(sub).value_counts().to_dict()}
  records.append(rec);print(rec,flush=True)
import json
json.dump(records,open('codex/research2_events.json','w'),indent=2)
