# own-phone fresh-SIM cell: can anything separate ATO from sim_swap labels? (5-fold LGB on raw + simple context)
from load8 import *
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
df,r=base()
acc=pd.read_csv(R+'data/upi-fraud-rings/accounts.csv')
df=df.merge(acc[['account_id','home_city','kyc_level','os']].rename(columns={'account_id':'sender_id'}),on='sender_id',how='left')
df['away']=(df.device_city!=df.home_city).astype(int); df['hr']=df.ts.dt.hour
df=df.sort_values('t'); g=df.groupby('sender_id')
df['n_s_1d']=g.t.transform(lambda s: s.rolling(1).count()) if False else g.t.transform('size')
df['dt_prev']=df.t-g.t.shift(1); df['dt_next']=g.t.shift(-1)-df.t
df['amt_bal']=df.amount/df.balance_before
df['su']=df.t/3600-df.sim_age_hours
c=df.own&(df.sim_age_hours<72)&(df.is_test==0)&df.label.isin(['account_takeover','sim_swap'])
x=df[c].copy()
# same SIM-change event: other rows of same sender with same implied SIM change (within 1h)
feats=['amount','balance_before','amt_bal','sim_age_hours','device_age_days','sender_account_age_days','new_payee','pin_retries','login_to_txn_sec','ip_risk_score','away','hr','dt_prev','dt_next','qr_vpa_mismatch']
for cc in ['txn_type','receiver_type','merchant_category','app','kyc_level','os']: x[cc]=x[cc].astype('category'); feats.append(cc)
y=(x.label=='account_takeover').astype(int).values
oof=np.zeros(len(x))
for trn,val in StratifiedKFold(5,shuffle=True,random_state=0).split(x,y):
    m=lgb.LGBMClassifier(n_estimators=200,learning_rate=.05,num_leaves=15,min_child_samples=20,verbose=-1,n_jobs=4)
    m.fit(x.iloc[trn][feats],y[trn]); oof[val]=m.predict_proba(x.iloc[val][feats])[:,1]
print('n',len(x),'ATO share',y.mean().round(3),'AUC',round(fauc(y,oof),3))
imp=pd.Series(m.feature_importances_,feats).sort_values(ascending=False); print(imp.head(8).to_dict())
for f in ['away','pin_retries','hr','sim_age_hours','amt_bal','new_payee','ip_risk_score']:
    print(f, x.groupby('label')[f].mean().round(3).to_dict())
