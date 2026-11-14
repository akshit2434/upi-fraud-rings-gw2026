from load import *
df=load()
df['ab']=df.groupby(['amount','balance_before']).txn_id.transform('size')
x=df[df.ab>1].sort_values(['amount','balance_before','t'])
print(df[df.ab>1].groupby(['label'],dropna=False).size())
cols=['timestamp','sender_id','receiver_id','device_id','amount','balance_before','txn_type','merchant_category','label','grp','is_test','sim_age_hours','login_to_txn_sec']
pd.set_option('display.width',250)
print(x[x.label.isin(['refund_abuse','phishing_collect','qr_swap','fake_cashback','loan_app_extortion'])|x.is_test.eq(1)][cols].head(40).to_string())
