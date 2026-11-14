import pandas as pd, numpy as np, sys
R='/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/'
sys.path.insert(0,R+'src'); sys.path.insert(0,'/tmp/hunt3')
from block_seg import *; from base import sep_pred
df=pd.read_parquet('/tmp/hunt5/raw.parquet'); df['dn']=df.device_id.str[2:].astype(int)
trall=df[df.is_test==0]; trb=trall[(trall.dn>=423000)&(trall.dn<428455)]
zt=trb[trb.label!='legit'].groupby('dn').label.agg(lambda s:s.value_counts().index[0])
s=trb.merge(sep_pred(),on='txn_id'); s['zone']=s.dn.map(zt); f=s[s.label!='legit']
logC,_=fit_emissions(f.label.values,f.sp.values,f.dn.values)
fl=trb[trb.label!='legit']; _,logPn=fit_emissions(fl.label.values,fl.label.values,fl.dn.values)
for pen in [0.5,1,2,3]:
    seg=segment(s.dn.values,s.sp.values,logC,logPn,switch_pen=pen); s['seg']=s.dn.map(seg); F=s[s.label!='legit']; FF=F[F.sp!='legit']
    print(pen, 'all fraud seg acc %.3f'%(F.seg==F.label).mean(), '| v6-fraud-called rows: v6 %.3f seg %.3f n=%d'%((FF.sp==FF.label).mean(),(FF.seg==FF.label).mean(),len(FF)))
np.save('/tmp/hunt5/logC.npy',logC); np.save('/tmp/hunt5/logPn.npy',logPn)
# Oct: v6 raw predictions
P=np.load(R+'preds/v6_R1_bag_test_P.npy'); w=np.load(R+'preds/v6_R1_bag_weights.npy'); ids=pd.read_csv(R+'preds/v6_R1_bag_test_ids.csv').iloc[:,0].values
v6=pd.DataFrame({'txn_id':ids,'v6':np.array(LABS)[(P*w).argmax(1)]})
v10=pd.read_csv(R+'submissions/v10_v6_octrules.csv').rename(columns={'id':'txn_id','label':'v10'})
te=df[df.is_test==1].merge(v6,on='txn_id').merge(v10,on='txn_id')
ob=te[(te.dn>=473632)&(te.dn<474552)].copy()
seg=segment(ob.dn.values,ob.v6.values,logC,logPn,switch_pen=2); ob['seg']=ob.dn.map(seg)
z=ob.groupby('seg').dn.agg(['min','max','size']).sort_values('min'); print(z)
print('changes vs v10:', (ob.seg!=ob.v10).sum()); print(pd.crosstab(ob.v10,ob.seg).to_string())
ob[['txn_id','dn','v6','v10','seg']].to_parquet('/tmp/hunt5/octblock_seg.parquet')
