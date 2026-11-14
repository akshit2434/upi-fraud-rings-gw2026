import pandas as pd, numpy as np, sys
sys.path.insert(0,'/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/research/hunt')
from load import load
df=load(); df['mk']=(df.txn_type!='QR')&df.qr_vpa_mismatch.notna()
df['m']=df.timestamp.str[:7]; df['iprk']=df.groupby('m').ip_risk_score.rank(pct=True)
df['dn']=df.device_id.str[2:].astype(int)
te=df[df.is_test==1]; tr=df[df.is_test==0]
sub=pd.read_csv('/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/submissions/v10_v6_octrules.csv').set_index('id').label
te=te.assign(p=te.txn_id.map(sub).values)
blk=(te.dn>=473632)&(te.dn<=474551)
grp={'test marker':te.mk,'test Oct devblock':blk,'test pred fraud':te.p!='legit','test pred legit':te.p=='legit',
     'test QRP2P blank':(te.txn_type=='QR')&(te.receiver_type=='P2P')&te.qr_vpa_mismatch.isna()}
for k,v in grp.items(): print(k, int(v.sum()), 'ip rank', round(te.iprk[v].mean(),3),'ip mean',round(te.ip_risk_score[v].mean(),3), 'ver', te.app_version[v].value_counts(normalize=True).sort_index().round(2).to_dict())
for k,v in {'Sep marker':tr.mk&(tr.m=='2026-09'),'Sep conn fraud':(tr.grp=='conn')&(tr.m=='2026-09'),'Sep iso phish':(tr.grp=='iso')&(tr.label=='phishing_collect')&(tr.m=='2026-09'),'Sep legit':(tr.label=='legit')&(tr.m=='2026-09')}.items():
    print(k,int(v.sum()),'ip rank',round(tr.iprk[v].mean(),3),'ip mean',round(tr.ip_risk_score[v].mean(),3))
print(np.percentile(te.ip_risk_score,[1,10,50,90,99,99.9]), np.percentile(tr[tr.m=='2026-09'].ip_risk_score,[1,10,50,90,99,99.9]))
