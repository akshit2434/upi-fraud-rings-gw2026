# raw-string format patterns per column: LP vs legit vs GEN vs test
from load8 import *
df,r=base()
g=df.g.values
cols=['timestamp','amount','sim_age_hours','device_age_days','sender_account_age_days','new_payee','pin_retries','login_to_txn_sec','balance_before','ip_risk_score','qr_vpa_mismatch','app_version','merchant_category','txn_type','receiver_type','app','device_city','txn_id','sender_id','receiver_id','device_id']
import re
def pat(s):
    # digit-class pattern: replace digits by 9, letters keep case class
    return s.str.replace(r'[0-9]','9',regex=True).str.replace(r'[a-f]','x',regex=True) if s.name in ('txn_id',) else s.str.replace(r'[0-9]','9',regex=True)
out=[]
for c in cols:
    s=r[c].astype(str)
    if c in ('merchant_category','txn_type','receiver_type','app','device_city'):
        p=s.str.len().astype(str)+'|'+s.str.contains(' ').astype(str)+'|'+(s!=s.str.strip()).astype(str)+'|'+s.str.isupper().astype(str)
    else:
        p=pat(s)
    t=pd.crosstab(p,g)
    t=t[t.sum(1)>=1]
    for k in ['legit','LP','GEN','test']: t[k+'%']=t[k]/t[k].sum()*100
    t['lift_LP']=t['LP%']/(t['legit%']+1e-9)
    print('=====',c,len(t),'patterns')
    print(t.sort_values('legit',ascending=False).head(12).round(3).to_string())
    odd=t[(t.LP>=5)&((t.lift_LP>1.5)|(t.lift_LP<0.67))]
    if len(odd): print('ODD',odd.round(3).to_string())
