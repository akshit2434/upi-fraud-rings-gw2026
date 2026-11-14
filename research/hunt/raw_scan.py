from load import *
df=load()
tr=df[df.is_test==0].copy()
acc=pd.read_csv(R+'data/upi-fraud-rings/accounts.csv'); mer=pd.read_csv(R+'data/upi-fraud-rings/merchants.csv')
tr=tr.merge(acc.add_prefix('s_'),left_on='sender_id',right_on='s_account_id',how='left')
tr=tr.merge(acc.add_prefix('r_'),left_on='receiver_id',right_on='r_account_id',how='left')
tr=tr.merge(mer.add_prefix('m_'),left_on='receiver_id',right_on='m_merchant_id',how='left')
def ndec(x, maxd=6):
    s=x.map(lambda v: '' if pd.isna(v) else repr(float(v)))
    return s.map(lambda z: len(z.split('.')[1].rstrip('0')) if '.' in z else 0)
F={}
for c in ['amount','balance_before','sim_age_hours','device_age_days','ip_risk_score','login_to_txn_sec','sender_account_age_days']:
    F[c+'_ndec']=ndec(tr[c])
    F[c+'_last']=((tr[c]*100).round()%10)
    F[c+'_int_last']=(tr[c].fillna(0).astype(float)//1)%10
    F[c+'_isna']=tr[c].isna().astype(int)
F['sec']=tr.ts.dt.second; F['minute']=tr.ts.dt.minute; F['hour']=tr.ts.dt.hour; F['dow']=tr.ts.dt.dayofweek; F['day']=tr.ts.dt.day
F['qr_isna']=tr.qr_vpa_mismatch.isna().astype(int)
F['mc_isna']=tr.merchant_category.isna().astype(int)
F['bal_lt_amt']=(tr.balance_before<tr.amount).astype(int)
F['amt_over_bal']=tr.amount/(tr.balance_before+1)
open_age=(tr.ts.dt.normalize()-pd.to_datetime(tr.s_opened_on)).dt.days
F['acct_age_diff']=tr.sender_account_age_days-open_age
F['dev_gt_acct']=(tr.device_age_days>tr.sender_account_age_days).astype(int)
F['dev_minus_acct']=tr.device_age_days-tr.sender_account_age_days
F['sim_h_vs_acct']=tr.sim_age_hours/24-tr.sender_account_age_days
F['away']=(tr.device_city!=tr.s_home_city).astype(int)
F['r_open_age']=(tr.ts.dt.normalize()-pd.to_datetime(tr.r_opened_on)).dt.days
F['r_kyc_min']=(tr.r_kyc_level!='full').astype(float).where(tr.r_kyc_level.notna())
F['s_kyc_min']=(tr.s_kyc_level!='full').astype(int)
F['same_bank']=(tr.s_bank_code==tr.r_bank_code).astype(float).where(tr.r_bank_code.notna())
F['r_city_eq']=(tr.r_home_city==tr.device_city).astype(float).where(tr.r_home_city.notna())
F['m_onb_age']=(tr.ts.dt.normalize()-pd.to_datetime(tr.m_onboarded_on)).dt.days
F['m_cat_mis']=(tr.m_category!=tr.merchant_category).astype(float).where(tr.m_category.notna())
F['m_city_eq']=(tr.m_city==tr.device_city).astype(float).where(tr.m_city.notna())
F['m_city_home']=(tr.m_city==tr.s_home_city).astype(float).where(tr.m_city.notna())
F['os_ios']=(tr.s_os=='ios').astype(int)
F['s_open_year']=pd.to_datetime(tr.s_opened_on).dt.year
F['dev_own']=(tr.device_id.str[2:]==tr.sender_id.str[2:]).astype(int)
F['dup_ts_sender']=tr.groupby(['sender_id','timestamp']).txn_id.transform('size')
F['dup_ts']=tr.groupby('timestamp').txn_id.transform('size')
F['dup_amt_day']=tr.groupby([tr.ts.dt.date,'amount']).txn_id.transform('size')
F=pd.DataFrame(F); F['grp']=tr.grp.values; F['label']=tr.label.values; F['txn_type']=tr.txn_type.values; F['rt']=tr.receiver_type.values
F.to_parquet('/tmp/hunt/raw_feats.parquet')
rows=[]
for cls in STRUCT+['phishing_collect','qr_swap','refund_abuse','loan_app_extortion','fake_cashback']:
    m=((F.label==cls)&(F.grp=='iso'))|(F.grp=='legit')
    sub=F[m]; y=(sub.label==cls).values
    mc=((F.label==cls)&(F.grp=='conn'))|(F.grp=='legit'); subc=F[mc]; yc=(subc.label==cls).values
    for c in F.columns[:-4]:
        a=auc(y,sub[c].values.astype(float)); ac=auc(yc,subc[c].values.astype(float))
        rows.append((cls,c,a,ac))
R_=pd.DataFrame(rows,columns=['cls','feat','auc_iso','auc_conn'])
R_['dev']=(R_.auc_iso-0.5).abs()
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
print(R_.sort_values('dev',ascending=False).head(60).to_string())
R_.to_csv('/tmp/hunt/raw_scan.csv',index=False)
