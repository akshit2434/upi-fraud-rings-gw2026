# AUC of each raw column: LP fraud (Sep, model-legit, non-block) vs model-legit legit; per class & pooled
from load import *
df=load(); m=sepval(df); s=df.merge(m,on='txn_id')
s['lp']=(s.label!='legit')&(~s.block)&(s.pf<0.05)
s=s[s.lp|((s.label=='legit')&(s.pf<0.05))].copy()
s['hour']=s.ts.dt.hour+s.ts.dt.minute/60
s['cents']=((s.amount*100).round()%100)
s['amt_int']=(s.amount==s.amount.round()).astype(int)
s['ab']=s.amount/(s.balance_before+1)
s['lamt']=np.log1p(s.amount)
s['qrnull']=s.qr_vpa_mismatch.isna().astype(int)
s['appn']=s.app.astype('category').cat.codes
s['appv']=s.app_version.astype(float)
cols=['amount','balance_before','ab','sim_age_hours','device_age_days','sender_account_age_days','new_payee','pin_retries','login_to_txn_sec','ip_risk_score','hour','cents','amt_int','own','qrnull','appn','appv']
y=s.lp.values
out={}
out['ALL']={c:fauc(y,s[c].astype(float)) for c in cols}
for cls in sorted(s[s.lp].label.unique()):
    mm=(s.label==cls)|(~s.lp)
    out[cls[:10]]={c:fauc(s.lp[mm],s.loc[mm,c].astype(float)) for c in cols}
pd.set_option('display.width',250)
print(pd.DataFrame(out).round(3))
