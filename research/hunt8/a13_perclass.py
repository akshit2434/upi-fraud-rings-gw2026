# per-class LP vs own-phone legit: KS / AUC on raw numerics + categorical TV distance
from load8 import *
df,r=base()
df['hr']=df.ts.dt.hour; df['sec']=df.ts.dt.second; df['dow']=df.ts.dt.dayofweek
df['cents']=np.round(df.amount*100)%100; df['bcents']=np.round(df.balance_before*100)%100
df['amt_bal']=df.amount/df.balance_before
df['mis']=df.qr_vpa_mismatch.fillna(-1)
tr=df[(df.is_test==0)&df.own&df.g.isin(['LP','legit'])]
L=tr[tr.g=='legit'].sample(300000,random_state=0)
num=['amount','balance_before','amt_bal','sim_age_hours','device_age_days','sender_account_age_days','login_to_txn_sec','pin_retries','new_payee','ip_risk_score','hr','sec','dow','cents','bcents','t']
cat=['txn_type','receiver_type','merchant_category','app','device_city','mis']
rows=[]
for c in sorted(tr[tr.g=='LP'].label.unique()):
    s=tr[(tr.g=='LP')&(tr.label==c)]
    d={'cls':c,'n':len(s)}
    for f in num:
        y=np.r_[np.ones(len(s)),np.zeros(len(L))]; v=np.r_[s[f].values,L[f].values]
        d[f]=round(fauc(y,v),3)
    for f in cat:
        a=s[f].astype(str).value_counts(normalize=True); b=L[f].astype(str).value_counts(normalize=True)
        d['TV_'+f]=round(0.5*(a.reindex(b.index,fill_value=0)-b).abs().sum(),3)
    rows.append(d)
o=pd.DataFrame(rows); pd.set_option('display.width',300)
print(o.to_string())
# null TV for a random legit sample of same size ~ 2000
s=tr[tr.g=='legit'].sample(2000,random_state=1)
print('null TV n=2000:',{f:round(0.5*(s[f].astype(str).value_counts(normalize=True).reindex(L[f].astype(str).value_counts().index,fill_value=0)-L[f].astype(str).value_counts(normalize=True)).abs().sum(),3) for f in cat})
