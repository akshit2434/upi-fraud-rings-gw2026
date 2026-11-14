import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
df=pd.read_parquet('/tmp/hunt5/rawj.parquet'); df['dn']=df.device_id.str[2:].astype(int); df['sn']=df.sender_id.str[2:].astype(int)
df['own']=df.dn==df.sn; df['away']=df.device_city!=df.s_home_city; df['hour']=df.ts.dt.hour
df['night']=df.hour.isin([0,1,2,3,4,5]); df['a2b']=df.amount/df.balance_before; df['fresh']=df.sim_age_hours<72
tr=df[(df.is_test==0)&df.conn&df.own&df.label.isin(['sim_swap','account_takeover'])]
print(tr.groupby('label')[['fresh','away','night','pin_retries','a2b','login_to_txn_sec','new_payee','sim_age_hours','device_age_days']].agg(['mean','median']).T.round(3).to_string())
y=(tr.label=='account_takeover').values
for c in ['fresh','away','night','pin_retries','a2b','login_to_txn_sec','new_payee','sim_age_hours','device_age_days','amount','balance_before']:
    print(c, round(roc_auc_score(y,tr[c].astype(float)),3))
print(pd.crosstab([tr.fresh,tr.away],tr.label))
