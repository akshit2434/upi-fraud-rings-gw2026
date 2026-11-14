from load import *
df=load(); tr=df[df.is_test==0]
tr=tr.assign(cat=tr.merchant_category.fillna('P2P'), hr=tr.ts.dt.hour, mon=tr.ts.dt.month, v3=tr.app_version.astype(str).str.startswith('3'))
leg=tr[tr.grp=='legit']
def tv(a,b): 
    a=a/a.sum(); b=b/b.sum(); idx=a.index.union(b.index); return 0.5*(a.reindex(idx,fill_value=0)-b.reindex(idx,fill_value=0)).abs().sum()
rows=[]
for cls in sorted(tr.label.unique()):
    if cls=='legit': continue
    for g in ['iso','conn']:
        s=tr[(tr.label==cls)&(tr.grp==g)]
        rows.append(dict(cls=cls,g=g,n=len(s),tv_cat=tv(s.cat.value_counts(),leg.cat.value_counts()),tv_type=tv(s.txn_type.value_counts(),leg.txn_type.value_counts()),
          tv_hr=tv(s.hr.value_counts(),leg.hr.value_counts()), tv_city=tv(s.device_city.value_counts(),leg.device_city.value_counts()),
          tv_app=tv(s.app.value_counts(),leg.app.value_counts()), v3=s.v3.mean(), mon7=(s.mon==7).mean(),mon9=(s.mon==9).mean(),
          np_=s.new_payee.mean(), pin=s.pin_retries.mean()))
rows.append(dict(cls='legit',g='',n=len(leg),v3=leg.v3.mean(),mon7=(leg.mon==7).mean(),mon9=(leg.mon==9).mean(),np_=leg.new_payee.mean(),pin=leg.pin_retries.mean()))
# baseline TV of random legit sample size 3000 vs legit
for n in [2000,5000]:
    s=leg.sample(n,random_state=1); rows.append(dict(cls=f'legit_sample{n}',g='',n=n,tv_cat=tv(s.cat.value_counts(),leg.cat.value_counts()),tv_type=tv(s.txn_type.value_counts(),leg.txn_type.value_counts()),tv_hr=tv(s.hr.value_counts(),leg.hr.value_counts()),tv_city=tv(s.device_city.value_counts(),leg.device_city.value_counts()),tv_app=tv(s.app.value_counts(),leg.app.value_counts()),v3=s.v3.mean()))
pd.set_option('display.width',250)
print(pd.DataFrame(rows).round(3).to_string())
