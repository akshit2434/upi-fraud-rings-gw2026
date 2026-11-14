import pandas as pd, numpy as np
df=pd.read_parquet('/tmp/hunt5/raw.parquet')
df['grp']=np.where(df.is_test==1,'test',np.where(df.label=='legit','legit',np.where(df.conn,'C','I')))
keys={'sender+amount':['sender_id','amount'],'amount+balance':['amount','balance_before'],'balance':['balance_before'],'sender+balance':['sender_id','balance_before'],
      'sim_age':['sim_age_hours','sender_id'],'login+amount':['login_to_txn_sec','amount','receiver_id'],'ts':['timestamp'],'ts+sender':['timestamp','sender_id'],'recv+amount':['receiver_id','amount'],'dev_age+sender':['device_age_days','sender_id']}
for k,c in keys.items():
    n=df.groupby(c).txn_id.transform('size')
    print(k.ljust(16), (n>1).groupby(df.grp).mean().round(4).to_dict())
