import pandas as pd, numpy as np
df=pd.read_parquet('/tmp/hunt5/raw.parquet'); df['dn']=df.device_id.str[2:].astype(int)
tr=df[(df.is_test==0)&(df.dn>=427840)&(df.dn<428460)].sort_values('dn')
ab={'refund_abuse':'R','fake_cashback':'F','loan_app_extortion':'L','merchant_collusion':'M','legit':'.','qr_swap':'Q','phishing_collect':'P','account_takeover':'A','sim_swap':'S','mule_account':'U','velocity_bot':'V','layering_chain':'Y'}
s=''.join(ab[l] for l in tr.label); print(len(tr)); 
for i in range(0,len(s),100): print(tr.dn.iloc[i], s[i:i+100])
# rows by train month in tail zones
tr['zone']=pd.cut(tr.dn,[427840,428060,428200,428302,428460],labels=['R','F','L','M'])
print(pd.crosstab(tr.zone,tr.timestamp.str[:7]))
