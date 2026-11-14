import pandas as pd, numpy as np, sys
from scipy.optimize import brentq
sys.path.insert(0,'/tmp/hunt3'); sys.path.insert(0,'/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/research/hunt')
from load import load
from base import sep_pred
df=load(); tr=df[df.is_test==0].copy(); tr['v3']=tr.app_version.astype(str).str.startswith('3')
sp=sep_pred(); s=tr.merge(sp,on='txn_id')
vL=s[s.label=='legit'].v3.mean()
sub=pd.read_csv('/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/submissions/v10_v6_octrules.csv').label.value_counts()
rows=[]
for c,x in s[s.label!='legit'].groupby('label'):
    hit=x[x.sp==c]; vG=hit.v3.mean()   # v3 rate of model-detectable (generated-profile) rows
    vC=x.v3.mean(); g=np.clip((vC-vL)/(vG-vL),0,1)
    r=(x.sp==c).mean()
    prec=(s[s.sp==c].label==c).mean()
    T=(tr.label==c).mean()*260582*1.115
    rows.append(dict(cls=c,v3_hit=vG,v3_class=vC,gen_frac=g,sep_recall=r,recall_of_gen=r/g if g>0 else np.nan,sep_prec=prec,T_test=T,pred_v10=sub.get(c,0)))
R=pd.DataFrame(rows).set_index('cls')
R['ceilF1']=2*R.gen_frac/(1+R.gen_frac)
print(R.round(3).to_string()); print('legit v3',round(vL,3))
print('ceiling macro perfect-on-gen:',round((0.985+R.ceilF1.sum())/12,4), ' gen share of fraud',round((R.gen_frac*R.T_test).sum()/R.T_test.sum(),3))
# test composition: gen_k from our v10 counts (prec/recall_of_gen from Sep); inj_k ~ Sep inj rate, scaled so total fraud = 4.08%
R['gen_test']=R.pred_v10*R.sep_prec/R.recall_of_gen.clip(upper=1)
inj_sep=(1-R.gen_frac)*R.T_test
scale=(0.0408*260582-R.gen_test.sum())/inj_sep.sum()
R['inj_test']=inj_sep*scale; R['T2']=R.gen_test+R.inj_test
R['ceil_test']=2*R.gen_test/(R.gen_test+R.T2)
R['now_test']=2*R.pred_v10*0.85/(R.pred_v10+R.T2)
print(R[['gen_test','inj_test','T2','ceil_test','now_test']].round(3).to_string())
print('inj scale',round(scale,3),'gen%',round(R.gen_test.sum()/260582*100,2),'inj%',round(R.inj_test.sum()/260582*100,2))
print('test ceiling macro', round((0.985+R.ceil_test.sum())/12,4),' est now',round((0.985+R.now_test.sum())/12,4))
