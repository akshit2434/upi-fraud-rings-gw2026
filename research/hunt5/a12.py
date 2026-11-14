import pandas as pd, numpy as np, sys
sys.path.insert(0,'/tmp/hunt3'); from base import sep_pred
df=pd.read_parquet('/tmp/hunt5/raw.parquet')
df['mk']=(df.txn_type!='QR')&df.qr_vpa_mismatch.notna()
df['v3']=df.app_version.astype(str).str.startswith('3')
s=df.merge(sep_pred(),on='txn_id')
print(len(s), 'legit v3', s[s.label=='legit'].v3.mean().round(3))
s['hit']=s.sp==s.label; s['fl']=s.sp!='legit'
r=s[s.label!='legit'].groupby(['label']).apply(lambda x:pd.Series({'n':len(x),'hit':x.hit.mean(),'v3_hit':x[x.hit].v3.mean(),'v3_wrongtype':x[x.fl&~x.hit].v3.mean(),'v3_miss':x[~x.fl].v3.mean(),'n_miss':(~x.fl).sum(),'ip_miss':x[~x.fl].ip_risk_score.mean(),'ip_hit':x[x.hit].ip_risk_score.mean()}))
r['gen_missed']=r.n_miss*(r.v3_miss-0.151)/(0.62-0.151)
print(r.round(3).to_string()); print('gen missed total',r.gen_missed.sum().round(0),'of fraud',r.n.sum(),'hits',(r.n*r.hit).sum().round(0))
print('legit ip',s[s.label=='legit'].ip_risk_score.mean())
# legit-labelled but flagged: v3?
print('legit-labelled flagged v3', s[(s.label=='legit')&s.fl].v3.mean(), (s[(s.label=='legit')&s.fl]).shape[0])
