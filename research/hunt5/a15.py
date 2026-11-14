import pandas as pd, numpy as np
df=pd.read_parquet('/tmp/hunt5/raw.parquet')
df['dn']=df.device_id.str[2:].astype(int)
df['own']=df.device_id.str[2:]==df.sender_id.str[2:]
df['zone']=np.select([df.own,(df.dn>=420000)&(df.dn<428455),(df.dn>=428455)&(df.dn<473632),(df.dn>=473632)&(df.dn<474552)],['own','trblock','tr2nd','octblock'],'oct2nd')
df['freshsim']=df.sim_age_hours<72; df['newdev']=df.device_age_days<2
tr=df[df.is_test==0]
print(pd.crosstab(tr.label,[tr.zone]).to_string())
print(pd.crosstab([tr.label,tr.conn],tr.zone).loc[['sim_swap','account_takeover','mule_account','layering_chain','velocity_bot']].to_string())
X=tr[tr.label.isin(['sim_swap','account_takeover'])]
print(X.groupby(['label','zone','conn']).agg(n=('txn_id','size'),fresh=('freshsim','mean'),newdev=('newdev','mean'),away=('device_city',lambda s:0)).round(2))
