from base import *
d=load(); d=d[d.grp!='aug'].copy(); d['lab']=np.where(d.grp=='oct',d.v9,d.label)
pd.set_option('display.width',250,'display.max_columns',30,'display.max_rows',300)
x=d[(d.txn_type=='COLLECT')].copy()
a=x.amount
x['pat']=np.select([a%1000==999,a%1000==990,a%500==499,a%100==99,a%100==0,(a.round(0)==a)],['x999','x990','x499','x99','r100','int'],'paise')
x['ab']=pd.cut(a,[0,1000,2000,5000,10000,25000,1e9]).astype(str)
def agg(g): return g.agg(n=('lab','size'),fr=('lab',lambda s:(s!='legit').mean()),ph=('lab',lambda s:(s=='phishing_collect').mean()))
t=agg(x.groupby(['pat','grp'])).unstack(1); print(t.round(3))
t=agg(x[x.pat.isin(['x999','x990','x499','x99'])].groupby(['ab','new_payee','grp'])).unstack(2); print(t.round(3))
# PAY P2P just-under too
y=d[(d.txn_type=='PAY')&(d.receiver_type=='P2P')].copy(); a=y.amount
y['pat']=np.select([a%1000==999,a%1000==990,a%500==499,a%100==99],['x999','x990','x499','x99'],'other')
print(agg(y.groupby(['pat','new_payee','grp'])).unstack(2).round(3))
