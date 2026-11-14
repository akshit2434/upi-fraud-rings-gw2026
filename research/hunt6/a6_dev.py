# LOO deviation from own entity profile + inter-arrival gaps (train full density + test)
from load import *
df=load()
d=df[['txn_id','sender_id','receiver_id','device_id','t','amount','balance_before','login_to_txn_sec','pin_retries','sim_age_hours','device_age_days','app','device_city','txn_type','merchant_category','receiver_type','is_test','mon']].copy()
d['hour']=(d.t/3600)%24
d['la']=np.log1p(d.amount); d['lb']=np.log1p(d.balance_before); d['ll']=np.log1p(d.login_to_txn_sec)
F=pd.DataFrame({'txn_id':d.txn_id})
def loo_mean(g,c):
    s=g[c].transform('sum'); n=g[c].transform('count')
    return (s-d[c])/(n-1).replace(0,np.nan), n
for k in ['sender_id','receiver_id']:
    g=d.groupby(k)
    for c in ['la','lb','ll','hour','pin_retries']:
        mu,n=loo_mean(g,c)
        if c=='hour':
            # circular
            d['hs']=np.sin(d.hour/24*2*np.pi); d['hc']=np.cos(d.hour/24*2*np.pi)
            ms,_=loo_mean(g,'hs'); mc,_=loo_mean(g,'hc')
            F[f'{k[:1]}_dev_hour']=-(ms*d.hs+mc*d.hc).values
        else:
            F[f'{k[:1]}_dev_{c}']=(d[c]-mu).abs().values
            F[f'{k[:1]}_sdev_{c}']=(d[c]-mu).values
    # categorical LOO share of same value
    for c in ['app','device_city','txn_type','merchant_category','device_id','receiver_id' if k=='sender_id' else 'sender_id']:
        if c==k: continue
        nk=g.txn_id.transform('size'); nkv=d.groupby([k,c],dropna=False).txn_id.transform('size')
        F[f'{k[:1]}_share_{c}']=((nkv-1)/(nk-1).replace(0,np.nan)).values
    # gaps
    o=d.sort_values([k,'t']); tt=o.t.values; kk=o[k].values
    prev=np.r_[np.nan,np.diff(tt)]; prev[np.r_[True,kk[1:]!=kk[:-1]]]=np.nan
    nxt=np.r_[np.diff(tt),np.nan]; nxt[np.r_[kk[1:]!=kk[:-1],True]]=np.nan
    F.loc[o.index,f'{k[:1]}_gprev']=np.log1p(prev); F.loc[o.index,f'{k[:1]}_gnext']=np.log1p(nxt)
    F.loc[o.index,f'{k[:1]}_gmin']=np.log1p(np.fmin(prev,nxt))
    F.loc[o.index,f'{k[:1]}_pos']=(o.groupby(k).cumcount()/o.groupby(k).t.transform('size')).values
F.to_parquet(C+'dev_feats.parquet')
m=sepval(df); s=df[['txn_id','label','block','grp']].merge(m,on='txn_id').merge(F,on='txn_id')
lp=(s.grp=='stealth')&(s.pf<0.05); ll=(s.label=='legit')&(s.pf<0.05)
x=s[lp|ll]; y=lp[lp|ll].values
cols=[c for c in F.columns if c!='txn_id']
r=pd.Series({c:fauc(y,x[c].astype(float)) for c in cols}).round(3)
pd.set_option('display.max_rows',200); print(r.sort_values())
# Jul-Aug: stealth own phone vs legit
s2=df[['txn_id','label','grp','mon','own']].merge(F,on='txn_id'); t=s2[s2.mon.isin([7,8])&s2.grp.isin(['legit','stealth'])]
print(pd.Series({c:fauc((t.grp=='stealth').values,t[c].astype(float)) for c in cols}).round(3).sort_values().to_string())
