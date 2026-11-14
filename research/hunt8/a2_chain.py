# per-sender chain relations of balance_before, sim_age, device_age, account_age (full density train+test)
from load8 import *
df,r=base()
df=df.sort_values(['sender_id','t']).reset_index(drop=True)
g=df.groupby('sender_id')
dt=(g.t.shift(-1)-df.t)
nb=g.balance_before.shift(-1)
df['bal_ratio']=nb/df.balance_before
df['bal_res']=nb-(df.balance_before-df.amount)
print('balance next/cur ratio quantiles by g'); print(df.groupby('g').bal_ratio.describe(percentiles=[.01,.1,.5,.9,.99]).round(3).to_string())
print('share exact bal_res==0', df.groupby('g').bal_res.apply(lambda x:(x.abs()<0.011).mean()).round(4).to_dict())
# account age consistency: aa - days since t0 should be constant per sender
df['aa0']=df.sender_account_age_days-np.floor(df.t/86400)
df['aa_dev']=df.aa0-g.aa0.transform('median')
print('acct-age deviation from sender median'); print(pd.crosstab(df.aa_dev.clip(-3,3),df.g,normalize='columns').round(4).to_string())
# device_age per (sender,device): implied first-use day
gd=df.groupby(['sender_id','device_id'])
df['fu']=df.t/86400-df.device_age_days
df['fu_dev']=df.fu-gd.fu.transform('median'); df['nsd']=gd.fu.transform('size')
m=df.nsd>=3
print('device first-use deviation (days), n>=3'); print(df[m].groupby('g').fu_dev.apply(lambda x:x.abs().describe(percentiles=[.5,.9,.99])).unstack().round(3).to_string())
df['su']=df.t/3600-df.sim_age_hours
df['su_dev']=df.su-gd.su.transform('median')
print('sim first-use deviation (hours), n>=3'); print(df[m].groupby('g').su_dev.apply(lambda x:x.abs().describe(percentiles=[.5,.9,.99])).unstack().round(3).to_string())
df[['txn_id','g','aa_dev','fu_dev','su_dev','nsd','bal_ratio']].to_parquet(C+'chain8.parquet')
