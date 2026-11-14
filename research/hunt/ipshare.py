from load import *
df=load(); tr=df[df.is_test==0].copy()
tr=tr[tr.ip_risk_score.notna()]
print('ip distinct',tr.ip_risk_score.nunique())
def nb_count(key, win, same_label):
    # count other rows with same key within +-win sec (optionally same label)
    k=[key,'label'] if same_label else [key]
    s=tr.sort_values(k+['t'])
    out=np.zeros(len(s))
    for _,g in s.groupby(k,sort=False):
        t=g.t.values
        if len(t)<2: continue
        lo=np.searchsorted(t,t-win); hi=np.searchsorted(t,t+win,side='right')
        out[s.index.get_indexer(g.index)]=hi-lo-1
    return pd.Series(out,index=s.index).reindex(tr.index)
import time; t0=time.time()
tr['ip_lab_1h']=nb_count('ip_risk_score',3600,True)
print(time.time()-t0)
tr['k']=np.where(tr.grp=='legit','legit',tr.label+'|'+tr.grp)
print(tr.groupby('k').ip_lab_1h.agg(lambda x:(x>0).mean()).round(4))
