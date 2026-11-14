from load import *
df=load()
tr=df[df.is_test==0].copy()
tr=tr.sort_values(['sender_id','t'])
g=tr.groupby('sender_id')
tr['nb']=g.balance_before.shift(-1); tr['pb']=g.balance_before.shift(1); tr['pa']=g.amount.shift(1)
tr['ndt']=g.t.shift(-1)-tr.t
tr['exp_next']=tr.balance_before-tr.amount
tr['res']=tr.nb-tr.exp_next
tr['res_prev']=tr.balance_before-(tr.pb-tr.pa)
print(tr[['balance_before','amount','nb','exp_next','res','ndt','label']].head(30).to_string())
for c in ['res','res_prev']:
    x=tr[c]
    print(c, 'exact-match share (|res|<0.01):', (x.abs()<0.01).mean(), 'ratio nb/bb median', (tr.nb/tr.balance_before).median())
