import pandas as pd, numpy as np
df=pd.read_parquet('/tmp/hunt5/raw.parquet')
tr=df[df.is_test==0].copy()
tr['grp']=np.where(tr.label=='legit','legit',np.where(tr.conn,'C','I'))
tr['isf']=(tr.label!='legit').astype(int)
tr['isc']=(tr.conn).astype(int)
# leave-one-out counts of fraud rows / connected fraud rows by entity
for k in ['sender_id','receiver_id','device_id']:
    g=tr.groupby(k)
    tr[k[:3]+'_nf']=g.isf.transform('sum')-tr.isf
    tr[k[:3]+'_nc']=g.isc.transform('sum')-tr.isc
    tr[k[:3]+'_n']=g.isf.transform('size')-1
tr['p2p']=tr.receiver_type=='P2P'
res=tr.groupby(['grp','p2p'])[[c for c in tr.columns if c.endswith('_nf') or c.endswith('_nc')]].apply(lambda x:(x>0).mean()).round(3)
print(res)
# per class isolated
res2=tr[tr.grp!='C'].groupby(['label','p2p'])[['rec_nf','rec_nc','sen_nf','dev_nf']].apply(lambda x:(x>0).mean()).round(3)
print(res2)
print(tr[tr.grp!='C'].groupby(['label','p2p'])[['rec_n','sen_n']].mean().round(1))
