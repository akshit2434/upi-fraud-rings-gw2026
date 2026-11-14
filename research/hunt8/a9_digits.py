# digit-level chi-square LP vs legit (own phones) on raw strings: last digit, first digit, length, digit after point
from load8 import *
from scipy.stats import chi2_contingency
df,r=base()
m=(df.is_test==0)&df.own&df.g.isin(['LP','legit'])
y=(df.g[m]=='LP').values
out=[]
for c in ['amount','balance_before','sim_age_hours','device_age_days','login_to_txn_sec','ip_risk_score','sender_account_age_days','timestamp','txn_id']:
    s=r[c][m].astype(str)
    for nm,feat in [('last',s.str[-1]),('first',s.str[0] if c!='txn_id' else s.str[2]),('len',s.str.len().astype(str)),('last2',s.str[-2:]),('pen',s.str[-2])]:
        t=pd.crosstab(feat.values,y)
        t=t[t.sum(axis=1)>=50]
        chi,p,dof,_=chi2_contingency(t.values)
        out.append((c,nm,len(t),round(chi,1),dof,p))
o=pd.DataFrame(out,columns=['col','feat','k','chi2','dof','p']).sort_values('p')
print(o.to_string())
