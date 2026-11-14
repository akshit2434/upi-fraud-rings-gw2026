import pandas as pd, numpy as np, sys
from scipy.stats import rankdata
STRUCT=['mule_account','velocity_bot','layering_chain','sim_swap','account_takeover','merchant_collusion']
F=pd.read_parquet(sys.argv[1])
def fauc(y,s):
    m=~np.isnan(s); y=y[m]; s=s[m]
    if y.sum()==0 or (~y).sum()==0: return np.nan
    r=rankdata(s); n1=y.sum(); n0=len(y)-n1
    return (r[y].sum()-n1*(n1+1)/2)/(n1*n0)
rng=np.random.default_rng(0)
leg=F[F.grp=='legit']; leg=leg.sample(200000,random_state=0)
feats=[c for c in F.columns if c not in ('grp','label','txn_type','rt','n','month')]
rows=[]
for cls in STRUCT+['phishing_collect','qr_swap','refund_abuse','loan_app_extortion','fake_cashback']:
    for g in ['iso','conn']:
        fr=F[(F.label==cls)&(F.grp==g)]
        sub=pd.concat([fr,leg]); y=(sub.label==cls).values
        for c in feats:
            rows.append((cls,g,c,fauc(y,sub[c].values.astype(float))))
R_=pd.DataFrame(rows,columns=['cls','g','feat','auc']).pivot_table(index=['cls','feat'],columns='g',values='auc').reset_index()
R_['dev']=(R_.iso-0.5).abs()
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
print(R_.sort_values('dev',ascending=False).head(70).round(3).to_string())
R_.to_csv(sys.argv[1]+'.auc.csv',index=False)
