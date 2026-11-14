from load import *
df=load()
keys={'s_amt':['sender_id','amount'],'s_bal':['sender_id','balance_before'],'s_sim':['sender_id','sim_age_hours'],'s_dev':['sender_id','device_age_days'],
 's_login':['sender_id','login_to_txn_sec'],'s_ip':['sender_id','ip_risk_score'],'sr_amt':['sender_id','receiver_id','amount'],'amt_bal':['amount','balance_before'],
 'r_amt':['receiver_id','amount'],'bal':['balance_before'],'sim':['sim_age_hours'],'s_r_day':['sender_id','receiver_id',df.ts.dt.date.rename('d')],
 's_min':['sender_id',df.ts.dt.floor('min').rename('m')],'r_min':['receiver_id',df.ts.dt.floor('min').rename('m')],'s_hour':['sender_id',df.ts.dt.floor('h').rename('h')]}
F=pd.DataFrame({'grp':df.grp,'label':df.label})
for k,cols in keys.items():
    F[k]=df.groupby(cols).txn_id.transform('size').values-1
F['k']=np.where(F.grp.isin(['legit','test']),F.grp,np.where(F.label.isin(STRUCT),'struct|'+F.grp,'single|'+F.grp))
pd.set_option('display.width',250)
print((F.groupby('k')[list(keys)].agg(lambda x:(x>0).mean())).round(4).to_string())
