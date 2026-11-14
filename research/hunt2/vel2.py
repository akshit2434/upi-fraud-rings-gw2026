from load import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
d=load(); d=d[~((d._is_test==0)&(~d._month.isin(['2026-08','2026-09'])))]
N=d.grp.value_counts()
x=d[(d.device_is_own==0)&d.merchant_category.isin(['ecommerce','gaming'])&(d.receiver_type=='MERCHANT')].copy()
x['lab']=np.where(x.grp=='oct',x.v6,x.label)
x['dn1h']=pd.cut(x.d_n_1h,[-1,1,2,4,1e9]).astype(str); x['dn1d']=pd.cut(x.s_n_1d,[-1,1,3,10,1e9]).astype(str)
x['nacc7']=pd.cut(x.d_nacc_p7d,[-1,1,2,5,1e9]).astype(str); x['login']=pd.cut(x.login_to_txn_sec,[-1,10,30,60,1e9]).astype(str)
x['devage']=pd.cut(x.device_age_days,[-1,1,7,60,1e9]).astype(str)
for f in ['dn1h','dn1d','nacc7','login','devage']:
    c=x.groupby([f,'grp']).size().unstack(fill_value=0)
    c=c.div(N[c.columns])*1e4
    vf=x[x.grp!='oct'].groupby(f).lab.apply(lambda s:(s=='velocity_bot').mean())
    v6=x[x.grp=='oct'].groupby(f).lab.apply(lambda s:(s!='legit').mean())
    print(pd.concat([c.round(2),vf.rename('trn_vel_share').round(3),v6.rename('oct_v6flag').round(3)],axis=1))
# velocity labels in train: how many are isolated (no device burst)?
t=d[d.grp!='oct']; v=t[t.label=='velocity_bot']
print('train velocity rows', len(v), 'own',v.device_is_own.mean().round(3),'d_n_1h>=2', (v.d_n_1h>=2).mean().round(3), 'ecom/gaming',v.merchant_category.isin(['ecommerce','gaming']).mean().round(3))
print('velocity per10k sep51', (d[d.grp=='sep'].label=='velocity_bot').mean()*1e4, 'v6 oct', (d[d.grp=='oct'].v6=='velocity_bot').mean()*1e4)
