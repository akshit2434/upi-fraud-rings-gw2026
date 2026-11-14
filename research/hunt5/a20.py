import pandas as pd, numpy as np, sys
sys.path.insert(0,'/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/src'); sys.path.insert(0,'/tmp/hunt3')
from block_seg import *; from base import sep_pred
df=pd.read_parquet('/tmp/hunt5/raw.parquet'); df['dn']=df.device_id.str[2:].astype(int)
tr=df[(df.is_test==0)&(df.dn>=423000)&(df.dn<428455)]
# true zone per device = majority non-legit label over ALL train months
zt=tr[tr.label!='legit'].groupby('dn').label.agg(lambda s:s.value_counts().index[0])
s=tr.merge(sep_pred(),on='txn_id')   # Sep rows (51% thinned frame) with v6 val predictions
s['zone']=s.dn.map(zt)
f=s[s.label!='legit']
# emissions: C from Sep block rows (lab=label), Pn from all-train devices
logC,logPn=fit_emissions(f.label.values,f.sp.values,f.dn.values)
_,logPn=fit_emissions(tr[tr.label!='legit'].label.values,tr[tr.label!='legit'].label.values,tr[tr.label!='legit'].dn.values)
for pen in [2,4,8]:
  for un in [True,False]:
    seg=segment(s.dn.values,s.sp.values,logC,logPn,switch_pen=pen,use_n=un)
    s['seg']=s.dn.map(seg)
    ok=s.zone.notna()
    print(f'pen {pen} use_n {un}: device-zone acc {(s[ok].seg==s[ok].zone).mean():.3f}', end=' | ')
    F=s[s.label!='legit']
    print(f'fraud rows: v6 class acc {(F.sp==F.label).mean():.3f}  seg acc {(F.seg==F.label).mean():.3f}  zone(oracle) acc {(F.zone==F.label).mean():.3f}  n={len(F)}')
seg=segment(s.dn.values,s.sp.values,logC,logPn,switch_pen=4)
s['seg']=s.dn.map(seg); F=s[s.label!='legit']
print(pd.crosstab(F.label,F.seg).to_string())
print('legit-labelled block rows in Sep:',(s.label=='legit').sum(),' v6 calls them', s[s.label=='legit'].sp.value_counts().head(5).to_dict())
