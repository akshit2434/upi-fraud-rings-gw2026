from load8 import *
df,r=base()
acc=pd.read_csv(R+'data/upi-fraud-rings/accounts.csv')
df=df.merge(acc[['account_id','opened_on']].rename(columns={'account_id':'sender_id'}),on='sender_id',how='left')
op=pd.to_datetime(df.opened_on)
df['d1']=(df.ts.dt.normalize()-op).dt.days
df['dfrac']=(df.ts-op).dt.total_seconds()/86400
df['aad']=df.sender_account_age_days-df.d1
print(pd.crosstab(df.aad.clip(-5,5),df.g,normalize='columns').round(4).to_string())
print(pd.crosstab(df.aad.clip(-5,5),df.g).to_string())
# is aad a function of hour?
df['hr']=df.ts.dt.hour
print(pd.crosstab(df.hr,df.aad.clip(-2,2),normalize='index').round(3).to_string())
# floor vs round of fractional days
for f,nm in [(np.floor,'floor'),(np.round,'round'),(np.ceil,'ceil')]:
    print(nm, df.groupby('g').apply(lambda x:(f(x.dfrac)==x.sender_account_age_days).mean()).round(4).to_dict())
df[['txn_id','aad']].to_parquet(C+'aad8.parquet')
