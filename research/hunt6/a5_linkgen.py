# do LP rows' entities ever touch generated fraud (conn/block, any time, any role)?
from load import *
df=load(); m=sepval(df); 
gen=df[df.grp.isin(['conn','block'])]
ents=set(gen.sender_id)|set(gen.receiver_id)
devs=set(gen.device_id)
gs=set(gen.sender_id); gr=set(gen.receiver_id)
s=df.merge(m,on='txn_id')
lp=(s.label!='legit')&(~s.block)&(s.pf<0.05); ll=(s.label=='legit')&(s.pf<0.05)
s['k']=np.where(lp,'LP',np.where(ll,'LL','other'))
s['s_gen']=s.sender_id.isin(ents); s['r_gen']=s.receiver_id.isin(ents); s['d_gen']=s.device_id.isin(devs)
s['s_gs']=s.sender_id.isin(gs); s['s_gr']=s.sender_id.isin(gr); s['r_gs']=s.receiver_id.isin(gs); s['r_gr']=s.receiver_id.isin(gr)
print(s.groupby('k')[['s_gen','r_gen','d_gen','s_gs','s_gr','r_gs','r_gr']].mean().round(4))
# also any fraud label (incl stealth) of other rows, excluding self
fr=df[(df.label.notna())&(df.label!='legit')]
sc=fr.sender_id.value_counts(); rc=fr.receiver_id.value_counts()
s['s_fr_any']=(s.sender_id.map(sc).fillna(0)+s.sender_id.map(rc).fillna(0)-(s.label!='legit'))>0
s['r_fr_any']=(s.receiver_id.map(sc).fillna(0)+s.receiver_id.map(rc).fillna(0)-(s.label!='legit'))>0
print(s.groupby('k')[['s_fr_any','r_fr_any']].mean().round(4))
print('--- LP split by conn flag')
s['k2']=s.k+'_'+s.grp
print(s[s.k!='other'].groupby('k2')[['s_gs','d_gen','r_gr']].agg(['mean','size']).round(4))
