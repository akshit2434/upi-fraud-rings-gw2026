# exact-value collisions: does an LP row share an exact continuous value with some generated-fraud row (or with any other row)?
from load import *
df=load(); m=sepval(df); s=df.merge(m,on='txn_id',how='left')
lp=(s.grp=='stealth')&(s.pf<0.05); ll=(s.label=='legit')&(s.pf<0.05)
gen=s[s.grp.isin(['conn','block'])]
for c in ['amount','balance_before','sim_age_hours','device_age_days','login_to_txn_sec','ip_risk_score','timestamp']:
    gv=set(gen[c].dropna().values)
    allc=s[c].value_counts()
    r={}
    for nm,mk in [('LP',lp),('LL',ll)]:
        v=s.loc[mk,c]
        r[nm+'_in_gen']=v.isin(gv).mean().round(4)
        r[nm+'_dup_any']=(v.map(allc)>1).mean().round(4)
    print(c,r)
