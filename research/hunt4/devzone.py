import pandas as pd, numpy as np, sys
sys.path.insert(0,'/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/research/hunt')
from load import load, STRUCT
df=load(); tr=df[df.is_test==0].copy(); te=df[df.is_test==1].copy()
for d in (tr,te):
    d['dn']=d.device_id.str[2:].astype(int); d['sn']=d.sender_id.str[2:].astype(int)
    d['zone']=np.select([d.dn==d.sn, d.dn<420000, (d.dn>=420000)&(d.dn<428455), (d.dn>=428455)&(d.dn<473632),(d.dn>=473632)&(d.dn<474552),d.dn>=474552],
                        ['own','other_acct','trainblock','train2nd','octblock','oct2nd'],'?')
tr['k']=np.where(tr.label=='legit','legit',np.where(tr.label.isin(STRUCT),'struct|','single|')+tr.grp)
print(pd.crosstab(tr.k,tr.zone).to_string()); print(pd.crosstab(tr.k,tr.zone,normalize='index').round(4).to_string())
print(te.zone.value_counts())
