# model-free 2-component mixture per class using two independent generator fingerprints (app v3, non-QR mismatch marker)
import pandas as pd, numpy as np, sys
from scipy.optimize import least_squares
sys.path.insert(0,'/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/research/hunt')
from load import load
df=load(); tr=df[(df.is_test==0)&(df.txn_type!='QR')].copy()
tr['v3']=tr.app_version.astype(str).str.startswith('3'); tr['mk']=tr.qr_vpa_mismatch.notna()
L=tr[tr.label=='legit']; vL=L.v3.mean(); mL3=L[L.v3].mk.mean(); mL4=L[~L.v3].mk.mean()
print('legit v3',round(vL,3),'mk|v3',round(mL3,4),'mk|v4',round(mL4,4))
rows=[]
for c,x in tr[tr.label!='legit'].groupby('label'):
    v=x.v3.mean(); m3=x[x.v3].mk.mean(); m4=x[~x.v3].mk.mean()
    def f(p):
        a,vG,mg=p
        p3=a*vG/(a*vG+(1-a)*vL); p4=a*(1-vG)/(a*(1-vG)+(1-a)*(1-vL))
        return [a*vG+(1-a)*vL-v, p3*mg+(1-p3)*mL3-m3, p4*mg+(1-p4)*mL4-m4]
    s=least_squares(f,[0.4,0.6,0.3],bounds=([0.01,0.16,0],[1,1,1]))
    rows.append(dict(cls=c,n=len(x),v3=v,mk_v3=m3,mk_v4=m4,gen_frac=s.x[0],vG=s.x[1],mk_gen=s.x[2],resid=np.abs(s.fun).max()))
print(pd.DataFrame(rows).set_index('cls').round(3).to_string())
