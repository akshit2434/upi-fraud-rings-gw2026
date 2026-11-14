from load import *
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
d=load(); d=d[~((d._is_test==0)&(~d._month.isin(['2026-08','2026-09'])))]
d['lab']=np.where(d.grp=='oct',d.v6,d.label)
def prof(s):
    return pd.Series({'n':len(s),'own':s.device_is_own.mean(),'away':s.away_from_home.mean(),'sim_med':s.sim_age_hours.median(),'simfresh':(s.sim_age_hours<72).mean(),
      'devage_med':s.device_age_days.median(),'pin':(s.pin_retries>0).mean(),'login_med':s.login_to_txn_sec.median(),'amt_med':s.amount.median(),
      'amt_bal_med':s.amt_over_bal.median(),'p2p':(s.receiver_type=='P2P').mean(),'ecom_gam':s.merchant_category.isin(['ecommerce','gaming']).mean(),
      'newp':s.new_payee.mean(),'senders':s.sender_id.nunique(),'devices':s.device_id.nunique(),'hr_med':s.hr.median()})
o=d[(d.grp=='oct')&(d.s_n_10m>=2)&(d.hr<6)]
print('Oct night sender-burst rows by v6'); print(o.groupby('v6').apply(prof).round(2).to_string())
t=d[(d.grp!='oct')&(d.s_n_10m>=2)]
print('Train (Aug+Sep51) sender-burst rows by label (any hour)'); print(t.groupby('label').apply(prof).round(2).to_string())
o2=d[(d.grp=='oct')&(d.s_n_10m>=2)&(d.hr>=6)]
print('Oct day sender-burst rows by v6'); print(o2.groupby('v6').apply(prof).round(2).to_string())
# connected sim_swap and ATO profiles in train (from all rows labelled, with sender burst or fresh sim)
for lab in ['sim_swap','account_takeover','velocity_bot']:
    s=d[(d.grp!='oct')&(d.label==lab)&((d.s_n_1h>=2)|(d.d_nacc_1h>=2))]
    print(lab,'connected-ish train', prof(s).round(2).to_dict())
