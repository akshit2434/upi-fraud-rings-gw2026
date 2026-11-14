# LP clustering per entity beyond independent selection (main path: own phone, non-mega merchant)
from load8 import *
df,r=base()
tr=df[(df.is_test==0)&(df.ts<'2026-10-01')&df.own&df.g.isin(['LP','legit'])].copy()
tr['y']=(tr.g=='LP').astype(int)
p=tr.y.mean(); print('p',p)
for key in ['sender_id','receiver_id','device_id']:
    a=tr.groupby(key).y.agg(['sum','size'])
    obs=(a['sum']*(a['sum']-1)/2).sum(); exp=(a['size']*(a['size']-1)/2*p*p).sum()
    # same-class pairs
    print(key,'LP pairs obs',int(obs),'exp',round(exp,1),'ratio',round(obs/exp,3))
lp=tr[tr.y==1]
a=lp.groupby(['sender_id','label']).size(); b=lp.groupby('sender_id').size()
print('same-sender LP pairs same class', int((a*(a-1)/2).sum()), 'of', int((b*(b-1)/2).sum()))
