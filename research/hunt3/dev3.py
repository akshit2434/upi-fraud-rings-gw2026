from base import *
te=pd.read_csv(R+'data/upi-fraud-rings/test.csv')
v9=pd.read_csv(R+'submissions/v9_v6_qrrule.csv').rename(columns={'id':'txn_id','label':'v9'}); te=te.merge(v9,on='txn_id')
pd.set_option('display.width',250,'display.max_columns',30,'display.max_rows',400)
te['dn']=te.device_id.str[2:].astype(int)
z=te[(te.dn>473631)&(te.dn<474900)].copy()
z['b']=(z.dn//25)*25
t=z.groupby('b').agg(n=('v9','size'),ndev=('device_id','nunique'),fr=('v9',lambda s:(s!='legit').mean()),
  top=('v9',lambda s:s[s!='legit'].value_counts().head(2).to_dict()),devage=('device_age_days','median'),sim=('sim_age_hours','median'),
  away=('device_city','size'),p2p=('receiver_type',lambda s:(s=='P2P').mean()),qr=('txn_type',lambda s:(s=='QR').mean()),amt=('amount','median'))
print(t.drop(columns='away').round(2).to_string())
z.to_parquet('newdev.parquet')
