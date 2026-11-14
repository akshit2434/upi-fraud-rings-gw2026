import pandas as pd, numpy as np
df=pd.read_parquet('/tmp/hunt5/raw.parquet')
tr=df[df.is_test==0]
p=tr[tr.receiver_type=='P2P'].copy(); p['rn']=p.receiver_id.str[2:].astype(int)
p['b']=p.rn//20000
print(pd.crosstab(p.b,p.label.where(p.label=='legit','fraud'),normalize='index').round(4).T.to_string())
c=p[p.conn&p.label.isin(['mule_account','layering_chain'])]
print('conn mule/layer receivers rn quantiles',c.rn.quantile([.01,.1,.5,.9,.99]).to_dict())
m=tr[tr.receiver_type=='MERCHANT'].copy(); m['mn']=m.receiver_id.str[2:].astype(int); m['b']=m.mn//1500
print(pd.crosstab(m.b,m.label.where(m.label=='legit','fraud'),normalize='index').round(4).T.to_string())
s=tr.copy(); s['sn']=s.sender_id.str[2:].astype(int); s['b']=s.sn//20000
print(pd.crosstab(s.b,s.label.where(s.label=='legit','fraud'),normalize='index').round(4).T.to_string())
