# per-row consistency of account age / device age / sim age with accounts.csv and with the account's other rows
from load import *
df=load()
acc=pd.read_csv(R+'data/upi-fraud-rings/accounts.csv')
df=df.merge(acc[['account_id','opened_on','home_city','kyc_level','os','bank_code']].rename(columns={'account_id':'sender_id'}),on='sender_id',how='left')
d=(df.ts.dt.normalize()-pd.to_datetime(df.opened_on)).dt.days
df['aad']=df.sender_account_age_days-d
print(pd.crosstab(df.aad.clip(-3,3),df.grp))
# device age vs time: implied first-use date
df['fu']=df.t/86400-df.device_age_days
df['su']=df.t/3600-df.sim_age_hours
for c in ['fu','su']:
    g=df.groupby(['sender_id','device_id'])[c]
    df[c+'_sd']=g.transform('std'); df[c+'_n']=g.transform('size')
    df[c+'_dev']=(df[c]-g.transform('median')).abs()
for c in ['fu_sd','su_sd','fu_dev','su_dev']:
    print(c, df[df.fu_n>=3].groupby('grp')[c].describe())
