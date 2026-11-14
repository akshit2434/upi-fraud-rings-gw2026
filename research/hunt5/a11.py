import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
df=pd.read_parquet('/tmp/hunt5/raw.parquet')
df['mk']=(df.txn_type!='QR')&df.qr_vpa_mismatch.notna()
df['month']=df.timestamp.str[:7]
df['v3']=df.app_version.astype(str).str.startswith('3')
for m,x in df.groupby('month'):
    if len(x)<1000: continue
    print(m, len(x), 'IP mk vs non', x[x.mk].ip_risk_score.describe()[['mean','50%']].round(3).tolist(), x[~x.mk].ip_risk_score.describe()[['mean','50%']].round(3).tolist(),
      'AUC', round(roc_auc_score(x.mk, x.ip_risk_score),3),
      'ver AUC(low ver=mk)', round(1-roc_auc_score(x.mk, x.app_version.astype(float)),3))
    if m<'2026-10': 
        f=x.label!='legit'; print('   label-fraud IP AUC',round(roc_auc_score(f,x.ip_risk_score),3), ' legit v3',x[~f].v3.mean().round(3),' fraud v3',x[f].v3.mean().round(3))
te=df[df.is_test==1]
print(te.groupby('app').app_version.value_counts().unstack())
print(te.groupby(['mk']).app_version.value_counts(normalize=True).unstack().round(3))
print(te.groupby(['mk']).app.value_counts(normalize=True).unstack().round(3))
tr=df[df.is_test==0]
print(tr.groupby(['mk']).app.value_counts(normalize=True).unstack().round(3))
