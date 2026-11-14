# model-free mixture on Sep rows (non-QR), plus v6 Sep recall split by fingerprint, -> recall on generated rows
import pandas as pd, numpy as np, sys
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hunt3'); sys.path.insert(0,'/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/research/hunt')
from load import load
from base import sep_pred
df=load(); df['v3']=df.app_version.astype(str).str.startswith('3'); df['mk']=df.qr_vpa_mismatch.notna()&(df.txn_type!='QR')
for mon in ['2026-07','2026-08','2026-09']:
    print(mon,'legit v3',round(df[(df.timestamp.str[:7]==mon)&(df.label=='legit')].v3.mean(),3),'fraud v3',round(df[(df.timestamp.str[:7]==mon)&(df.label.notna())&(df.label!='legit')].v3.mean(),3))
sp=sep_pred(); s=df.merge(sp,on='txn_id')
def solve(x,L):
    vL=L.v3.mean(); mL3=L[L.v3].mk.mean(); mL4=L[~L.v3].mk.mean()
    v=x.v3.mean(); m3=x[x.v3].mk.mean(); m4=x[~x.v3].mk.mean()
    def f(p):
        a,vG,mg=p; p3=a*vG/(a*vG+(1-a)*vL); p4=a*(1-vG)/(a*(1-vG)+(1-a)*(1-vL))
        return [a*vG+(1-a)*vL-v, p3*mg+(1-p3)*mL3-m3, p4*mg+(1-p4)*mL4-m4]
    return least_squares(f,[0.4,0.6,0.3],bounds=([0.01,0.16,0],[1,1,1])).x
rows=[]
nq=s[s.txn_type!='QR']; L=nq[nq.label=='legit']
for c,x in nq[nq.label!='legit'].groupby('label'):
    a,vG,mg=solve(x,L)
    # recall on gen: model hits among class rows / gen rows ; also estimate hits on legit-like part via v4&unmarked
    r=(x.sp==c).mean()
    # direct: P(hit|v3) and P(hit|v4) -> solve rho_gen, rho_like: P(hit|v3)=p3*rg+(1-p3)*rl
    vL=L.v3.mean(); p3=a*vG/(a*vG+(1-a)*vL); p4=a*(1-vG)/(a*(1-vG)+(1-a)*(1-vL))
    h3=(x[x.v3].sp==c).mean(); h4=(x[~x.v3].sp==c).mean()
    A=np.array([[p3,1-p3],[p4,1-p4]]); rg,rl=np.linalg.solve(A,[h3,h4])
    rows.append(dict(cls=c,n=len(x),gen_frac=a,vG=vG,mk_gen=mg,recall=r,recall_gen=rg,recall_like=rl))
R=pd.DataFrame(rows).set_index('cls'); print(R.round(3).to_string())
R['ceil']=2*R.gen_frac/(1+R.gen_frac); print('Sep non-QR ceiling macro (perfect on gen):',round((0.985+R.ceil.sum())/12,3))
