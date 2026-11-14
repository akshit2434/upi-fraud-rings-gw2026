from base import *
pd.set_option('display.width',250,'display.max_columns',30,'display.max_rows',400)
tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv'); te=pd.read_csv(R+'data/upi-fraud-rings/test.csv')
v9=pd.read_csv(R+'submissions/v9_v6_qrrule.csv').rename(columns={'id':'txn_id','label':'v9'}); te=te.merge(v9,on='txn_id')
s=pd.read_parquet('sepp.parquet')[['txn_id','sp']]
for x in (tr,te): x['dn']=x.device_id.str[2:].astype(int); x['own']=x.device_id.str[3:]==x.sender_id.str[2:]
tr=tr.merge(s,on='txn_id',how='left')
B_tr=tr[(tr.dn>=420000)&(tr.dn<428455)]; B_te=te[(te.dn>=473632)&(te.dn<474552)]
L_tr=tr[(tr.dn>=428455)]; L_te=te[te.dn>=474552]
def prof(x):
    return pd.Series(dict(n=len(x),qr=(x.txn_type=='QR').mean(),p2p=(x.receiver_type=='P2P').mean(),amt=x.amount.median(),
        paise=(x.amount.round(2)!=x.amount.round(0)).mean(),mk=((x.txn_type!='QR')&x.qr_vpa_mismatch.notna()).mean(),
        devage=x.device_age_days.median(),sim=x.sim_age_hours.median(),np_=x.new_payee.mean(),login=x.login_to_txn_sec.median(),
        own=x.own.mean(),rows_per_dev=len(x)/x.device_id.nunique(),old_app=(x.app_version<4).mean() if 'app_version' in x else np.nan))
r={'train block label=legit':prof(B_tr[B_tr.label=='legit']),'train block fraud':prof(B_tr[B_tr.label!='legit']),
   'Sep block v6-legit (sampled)':prof(B_tr[B_tr.sp=='legit']),
   'Oct block v9-legit':prof(B_te[B_te.v9=='legit']),'Oct block v9-fraud':prof(B_te[B_te.v9!='legit']),
   'train legit 2nd-phone block':prof(L_tr),'Oct legit 2nd-phone block':prof(L_te)}
print(pd.DataFrame(r).T.round(3).to_string())
print('Oct block',len(B_te),'v9 legit',(B_te.v9=='legit').sum()); print('train block legit share by month',B_tr.groupby(B_tr.timestamp.str[:7]).label.apply(lambda s:(s=='legit').mean()).round(3).to_dict())
print('Oct app_version block legit vs fraud', B_te.groupby(B_te.v9=='legit').app_version.value_counts(normalize=True).round(2).unstack())
print('Oct 2nd-phone block app',L_te.app_version.value_counts(normalize=True).round(2).to_dict())
