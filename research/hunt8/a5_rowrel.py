# exact within-row relations by group
from load8 import *
df,r=base()
acc=pd.read_csv(R+'data/upi-fraud-rings/accounts.csv')
df=df.merge(acc[['account_id','opened_on']].rename(columns={'account_id':'sender_id'}),on='sender_id',how='left')
op=pd.to_datetime(df.opened_on)
R_={}
R_['amt>bal']=df.amount>df.balance_before
R_['amt==bal']=(df.amount-df.balance_before).abs()<0.005
R_['amt>=.9bal']=df.amount>=.9*df.balance_before
R_['sim>dev*24']=df.sim_age_hours>df.device_age_days*24
R_['sim>acct*24']=df.sim_age_hours>df.sender_account_age_days*24
R_['dev>acct']=df.device_age_days>df.sender_account_age_days
R_['dev>acct+1']=df.device_age_days>df.sender_account_age_days+1
R_['dev_age<since_jul']=df.device_age_days<df.t/86400
R_['sim<since_jul']=df.sim_age_hours/24<df.t/86400
R_['dev_int']=(df.device_age_days*10%10==0)
R_['sim_int']=(df.sim_age_hours*10%10==0)
R_['amt_int']=(df.amount%1==0)
R_['bal_int']=(df.balance_before%1==0)
R_['bal_round100']=(df.balance_before%100==0)
R_['login<=5']=df.login_to_txn_sec<=5
R_['ip==0']=df.ip_risk_score==0
R_['ip>=.999']=df.ip_risk_score>=.999
R_['sec==0']=df.ts.dt.second==0
R_['minsec==0']=(df.ts.dt.second==0)&(df.ts.dt.minute==0)
R_['ts_before_open']=df.ts<op
R_['sim0']=df.sim_age_hours==0
R_['dev0']=df.device_age_days==0
R_['bal<amt*1.0001&>amt']=(df.balance_before>df.amount)&(df.balance_before<df.amount*1.01)
X=pd.DataFrame(R_); X['g']=df.g.values
t=X.groupby('g').mean().T*100
t['lift']=t.LP/t.legit
print(t.round(4).to_string())
