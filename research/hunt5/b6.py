import pandas as pd, numpy as np
df=pd.read_parquet('/tmp/hunt5/raw.parquet'); df['dn']=df.device_id.str[2:].astype(int)
tr=df[df.is_test==0].copy(); tr['blk']=(tr.dn>=420000)&(tr.dn<428455)
b=tr[tr.blk&(tr.label!='legit')][['sender_id','t','label','receiver_id']]
# other rows (non-block) of senders who have a block fraud row; time gap to nearest block row of same sender
o=tr[~tr.blk][['txn_id','sender_id','t','label','receiver_id']].merge(b.rename(columns={'t':'tb','label':'lb','receiver_id':'rb'}),on='sender_id')
o['gap']=(o.t-o.tb).abs()/3600
o=o.sort_values('gap').drop_duplicates('txn_id')
o['gb']=pd.cut(o.gap,[-1,1,6,24,24*7,24*30,1e9])
print(o.groupby('gb').apply(lambda x:pd.Series({'n':len(x),'fraud':(x.label!='legit').mean(),'same_cls':(x.label==x.lb).mean()})).round(3))
print('base fraud rate non-block', (tr[~tr.blk].label!='legit').mean().round(4))
# receivers of block rows: their incoming rows from non-block
r=tr[~tr.blk][['txn_id','receiver_id','t','label']].merge(b[['receiver_id','t','label']].rename(columns={'t':'tb','label':'lb'}),on='receiver_id')
r=r[r.receiver_id.str.startswith('AC')]; r['gap']=(r.t-r.tb).abs()/3600; r=r.sort_values('gap').drop_duplicates('txn_id'); r['gb']=pd.cut(r.gap,[-1,1,6,24,24*7,1e9])
print('P2P receivers of block rows -> other incoming'); print(r.groupby('gb').apply(lambda x:pd.Series({'n':len(x),'fraud':(x.label!='legit').mean(),'same':(x.label==x.lb).mean()})).round(3))
