import pandas as pd, numpy as np, sys
R='/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/'
sys.path.insert(0,R+'src'); sys.path.insert(0,'/tmp/hunt3')
import importlib, block_seg; importlib.reload(block_seg); from block_seg import *; from base import sep_pred
df=pd.read_parquet('/tmp/hunt5/raw.parquet'); df['dn']=df.device_id.str[2:].astype(int)
trall=df[df.is_test==0]; trb=trall[(trall.dn>=423000)&(trall.dn<428455)]
zt=trb[trb.label!='legit'].groupby('dn').label.agg(lambda s:s.value_counts().index[0])
logC=np.load('/tmp/hunt5/logC.npy'); logPn=np.load('/tmp/hunt5/logPn.npy')
s=trb.merge(sep_pred(),on='txn_id'); s['zone']=s.dn.map(zt)
# profile table from NON-Sep train block rows (Jul-Aug) to avoid in-sample
ja=trb[trb.timestamp<'2026-09-01']; ja=ja.assign(zone=ja.dn.map(zt)); ja=ja[ja.zone.notna()]
logT=fit_profile(profile_key(ja),ja.zone.values)
RL=profile_ll(profile_key(s),logT)
for pen in [1,2]:
  for w in [0,0.5,1.0]:
    seg=segment(s.dn.values,s.sp.values,logC,logPn,switch_pen=pen,row_ll=RL,row_w=w); s['seg']=s.dn.map(seg); F=s[s.label!='legit']; FF=F[F.sp!='legit']
    print(pen,w,'all fraud seg acc %.3f'%(F.seg==F.label).mean(),' fraud-called %.3f'%(FF.seg==FF.label).mean(), ' velocity rows right', ((F.label=='velocity_bot')&(F.seg=='velocity_bot')).sum(),'/',(F.label=='velocity_bot').sum())
seg=segment(s.dn.values,s.sp.values,logC,logPn,switch_pen=1,row_ll=RL,row_w=1.0); s['seg']=s.dn.map(seg); F=s[s.label!='legit']
print(pd.crosstab(F.label,F.seg).to_string())
