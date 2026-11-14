import pandas as pd, numpy as np
df=pd.read_parquet('/tmp/hunt5/raw.parquet'); df['dn']=df.device_id.str[2:].astype(int)
trb=df[(df.is_test==0)&(df.dn>=423000)&(df.dn<428455)]
zt=trb[trb.label!='legit'].groupby('dn').label.agg(lambda s:s.value_counts().index[0])
trb=trb.assign(zone=trb.dn.map(zt))
trb['tt']=trb.txn_type+'_'+trb.receiver_type
pd.set_option('display.width',250)
print(pd.crosstab(trb.zone,trb.tt,normalize='index').round(2).to_string())
print(trb.groupby('zone').agg(amt=('amount','median'),cat_na=('merchant_category',lambda s:s.isna().mean()),mis=('qr_vpa_mismatch',lambda s:s.notna().mean()),dage=('device_age_days','median'),sim=('sim_age_hours','median'),int_amt=('amount',lambda s:(s%1==0).mean())).round(2).to_string())
