# new_payee vs observed pair history (full-density train timeline): consistency for LP vs legit
from load import *
df=load()
d=df[df.is_test==0][['txn_id','sender_id','receiver_id','t','new_payee','label','grp']].sort_values('t')
d['k']=d.groupby(['sender_id','receiver_id']).cumcount()   # prior payments seen to this receiver
d['prior']=(d.k>0).astype(int)
m=sepval(df); s=d.merge(m,on='txn_id')
lp=(s.grp=='stealth')&(s.pf<0.05); ll=(s.label=='legit')&(s.pf<0.05)
s['g']=np.where(lp,'LP',np.where(ll,'LL','x'))
print(pd.crosstab([s.g,s.prior],s.new_payee,normalize='index').round(3))
print(pd.crosstab([s.g],s.prior,normalize='index').round(3))
# all train by group (no model)
print(pd.crosstab([d.grp,d.prior],d.new_payee,normalize='index').round(3))
