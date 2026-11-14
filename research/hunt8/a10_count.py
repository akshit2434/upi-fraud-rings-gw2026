# relabel vs insertion: sender row-count of LP rows vs legit rows (own phone, train Jul-Sep)
from load8 import *
df,r=base()
tr=df[(df.is_test==0)&(df.ts<'2026-10-01')]
n=tr.groupby('sender_id').size(); tr=tr.assign(n=tr.sender_id.map(n))
nm=tr.groupby(['sender_id',tr.mon]).size()
tr['nmon']=pd.Series(list(zip(tr.sender_id,tr.mon))).map(nm).values
for col in ['n','nmon']:
    print(col, tr[tr.own].groupby('g')[col].describe(percentiles=[.1,.25,.5,.75,.9]).round(2).to_string())
# excluding own row: n-1 for LP vs legit — under relabel identical
x=tr[tr.own&tr.g.isin(['LP','legit'])]
print(pd.crosstab(x.n.clip(0,20),x.g,normalize='columns').round(4).to_string())
# per-account: accounts with LP rows, distribution of n_legit (rows labelled legit)
