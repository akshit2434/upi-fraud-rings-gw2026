import pandas as pd, numpy as np
df=pd.read_parquet('/tmp/hunt5/raw.parquet')
df['grp']=np.where(df.is_test==1,'test',np.where(df.label=='legit','legit',np.where(df.conn,'C','I')))
k=['amount','balance_before']
df['n']=df.groupby(k).txn_id.transform('size')
d=df[df.n==2].sort_values(k+['t'])
a=d.iloc[::2].reset_index(drop=True); b=d.iloc[1::2].reset_index(drop=True)
assert (a.amount.values==b.amount.values).all()
pairs=pd.DataFrame({'ga':a.grp,'gb':b.grp,'la':a.label,'lb':b.label,'dt_h':(b.t-a.t)/3600,'same_sender':a.sender_id.values==b.sender_id.values,'same_recv':a.receiver_id.values==b.receiver_id.values,
  'same_dev':a.device_id.values==b.device_id.values,'same_type':a.txn_type.values==b.txn_type.values,'same_cat':(a.merchant_category.values==b.merchant_category.values),'amt':a.amount,'bal':a.balance_before,
  'same_city':a.device_city.values==b.device_city.values,'same_sim':a.sim_age_hours.values==b.sim_age_hours.values,'same_login':a.login_to_txn_sec.values==b.login_to_txn_sec.values,'same_ip':a.ip_risk_score.values==b.ip_risk_score.values})
pairs['pg']=[ '/'.join(sorted([x,y])) for x,y in zip(pairs.ga,pairs.gb)]
print(pairs.pg.value_counts().head(15))
print(pairs.groupby('pg')[['same_sender','same_recv','same_dev','same_type','same_cat','same_city','same_sim','same_login','same_ip']].mean().round(2).to_string())
print(pairs.groupby('pg').dt_h.describe().round(2).to_string())
print(pairs[pairs.pg.str.contains('I')].head(20).to_string())
