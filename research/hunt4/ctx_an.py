import pandas as pd, numpy as np, sys
sys.path.insert(0,'/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/research/hunt')
from load import fauc
pd.set_option('display.width',250)
S={n:pd.read_parquet(f'/tmp/hunt4/ctx_{n}.parquet') for n in ['full','U','L','test']}
cols=[c for c in S['full'].columns if c not in ('grp','lab','struct')]
rows=[]
for c in cols:
    r={'stat':c}
    for n in ['full','U','L']:
        F=S[n]; m=(F.grp=='legit')|((F.grp=='iso')&F.struct); y=(F.grp[m]=='iso').values
        r['auc_'+n]=fauc(y,F[c][m].values)
        m2=(F.grp=='legit')|(F.grp=='iso')&~F.struct; r['aucSingle_'+n]=fauc((F.grp[m2]=='iso').values,F[c][m2].values)
        r['mean_'+n]=F[c].mean()
    r['mean_test']=S['test'][c].mean()
    rows.append(r)
R=pd.DataFrame(rows).set_index('stat').round(3); print(R.to_string())
