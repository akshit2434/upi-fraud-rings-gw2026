from base import *
tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv')
pd.set_option('display.width',250,'display.max_columns',30,'display.max_rows',400)
tr['dn']=tr.device_id.str[2:].astype(int)
f=tr[(tr.dn>=420000)&(tr.dn<428500)].copy(); f['b']=(f.dn//100)*100
t=f.groupby('b').agg(n=('label','size'),ndev=('device_id','nunique'),fr=('label',lambda s:(s!='legit').mean()),top=('label',lambda s:s[s!='legit'].value_counts(normalize=True).head(2).round(2).to_dict()),devage=('device_age_days','median'))
print(t.to_string())
# per month: which ids used per month?
f['mon']=f.timestamp.str[:7]; print(f.groupby(['mon']).dn.agg(['min','max','nunique']))
