import pandas as pd, numpy as np
from scipy.stats import chisquare
df=pd.read_parquet('/tmp/hunt5/raw.parquet'); tr=df[df.is_test==0].copy()
tr['grp']=np.where(tr.label=='legit','legit',np.where(tr.conn,'C','I'))
S7=['mule_account','velocity_bot','layering_chain','sim_swap','account_takeover','merchant_collusion']
I=tr[(tr.grp=='I')&tr.label.isin(S7)]; L=tr[tr.grp=='legit']
for nm,f in [('hour',lambda x:x.ts.dt.hour),('sec',lambda x:x.ts.dt.second),('min',lambda x:x.ts.dt.minute),('wd',lambda x:x.ts.dt.weekday),('day',lambda x:x.ts.dt.day)]:
    a=f(I).value_counts().sort_index(); b=f(L).value_counts(normalize=True).sort_index()
    exp=b.reindex(a.index).fillna(1e-9)*a.sum(); st=chisquare(a,exp)
    r=(a/exp)
    print(nm,'chi2 p=%.3g'%st.pvalue,'max ratio %.2f min %.2f'%(r.max(),r.min()), r.round(2).to_dict() if nm in ['hour','wd'] else '')
