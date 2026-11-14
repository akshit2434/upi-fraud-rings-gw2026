from base import *
d=load(); d['mk']=(d.txn_type!='QR')&d.qr_vpa_mismatch.notna()
O=d[d.grp=='oct']
pd.set_option('display.width',250,'display.max_columns',30,'display.max_rows',200)
cols=['timestamp','sender_id','receiver_id','device_id','amount','txn_type','cat','sim_age_hours','device_age_days','login_to_txn_sec','new_payee','pin_retries','mk','v9','pfraud']
x=O[(O.hr.between(5,8))&(O.device_is_own==0)&O.mk]; print(x.v9.value_counts()); print(x.groupby('device_id').size().sort_values().tail(10))
x=O[(O.cat=='lending')]; print('lending oct', len(x), x.mk.sum(), x.v9.value_counts().to_dict())
S=d[d.grp=='sep']; y=S[S.cat=='lending']; print('lending sep51',len(y), y.mk.sum(), y.label.value_counts().to_dict())
print(O[(O.cat=='lending')&O.mk][cols].sort_values('timestamp').to_string())
