import pandas as pd, numpy as np, sys
sys.path.insert(0,'/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/research/hunt')
from load import load
df=load(); te=df[df.is_test==1].copy(); sep=df[(df.is_test==0)&(df.timestamp.str[:7]=='2026-09')]
sub=pd.read_csv('/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/submissions/v10_v6_octrules.csv').set_index('id').label
te['p']=te.txn_id.map(sub)
bins=[0,0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.2,0.3,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.8,1.01]
a=pd.cut(te.ip_risk_score,bins,right=False).value_counts(sort=False)
b=pd.cut(sep.ip_risk_score,bins,right=False).value_counts(sort=False)
bl=pd.cut(sep.ip_risk_score[sep.label=='legit'],bins,right=False).value_counts(sort=False)
fr=sep.groupby(pd.cut(sep.ip_risk_score,bins,right=False),observed=False).label.apply(lambda s:(s!='legit').mean())
tf=te.groupby(pd.cut(te.ip_risk_score,bins,right=False),observed=False).p.apply(lambda s:(s!='legit').mean())
print(pd.DataFrame({'test':a,'sep':b,'sep_legit':bl,'sep_fraud_rate':fr.round(3),'test_predfraud':tf.round(3)}).to_string())
print('test ip decimals', te.ip_risk_score.map(lambda v: len(repr(v).split('.')[1]) if '.' in repr(v) else 0).value_counts().head().to_dict())
print('sep ip decimals', sep.ip_risk_score.map(lambda v: len(repr(v).split('.')[1]) if '.' in repr(v) else 0).value_counts().head().to_dict())
