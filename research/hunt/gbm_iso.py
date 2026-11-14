from load import *
import lightgbm as lgb
df=load()
acc=pd.read_csv(R+'data/upi-fraud-rings/accounts.csv'); mer=pd.read_csv(R+'data/upi-fraud-rings/merchants.csv')
df=df.merge(acc.add_prefix('s_'),left_on='sender_id',right_on='s_account_id',how='left').merge(acc.add_prefix('r_'),left_on='receiver_id',right_on='r_account_id',how='left').merge(mer.add_prefix('m_'),left_on='receiver_id',right_on='m_merchant_id',how='left')
X=pd.DataFrame(index=df.index)
for c in ['amount','sim_age_hours','device_age_days','sender_account_age_days','new_payee','pin_retries','login_to_txn_sec','balance_before','qr_vpa_mismatch']:
    X[c]=df[c]
for c in ['receiver_type','txn_type','merchant_category','device_city','app','s_home_city','s_bank_code','s_kyc_level','s_os','r_kyc_level','r_bank_code','r_os','m_category','m_city']:
    X[c]=df[c].astype('category')
X['hour']=df.ts.dt.hour; X['dow']=df.ts.dt.day_of_week; X['cents']=(df.amount*100).round()%100; X['amt_mod100']=df.amount%100
X['own_dev']=(df.device_id.str[3:]==df.sender_id.str[2:]).astype(int)
X['away']=(df.device_city!=df.s_home_city).astype(int)
X['r_open_age']=(df.ts-pd.to_datetime(df.r_opened_on)).dt.days; X['m_onb']=(df.ts-pd.to_datetime(df.m_onboarded_on)).dt.days
X['amt_bal']=df.amount/(df.balance_before+1)
for k in ['sender_id','receiver_id','device_id']:
    X['n_'+k]=df.groupby(k).txn_id.transform('size'); X['nm_'+k]=df.groupby([k,df.ts.dt.month]).txn_id.transform('size')
X['pair_n']=df.groupby(['sender_id','receiver_id']).txn_id.transform('size')
X['s_nrecv']=df.groupby('sender_id').receiver_id.transform('nunique'); X['r_nsend']=df.groupby('receiver_id').sender_id.transform('nunique')
X['d_nsend']=df.groupby('device_id').sender_id.transform('nunique')
X['s_as_r']=df.sender_id.map(df.receiver_id.value_counts()).fillna(0); X['r_as_s']=df.receiver_id.map(df.sender_id.value_counts()).fillna(0)
X['s_hour_n']=df.groupby(['sender_id',df.ts.dt.floor('h')]).txn_id.transform('size')
mon=df.ts.dt.month
y=df.label.isin(STRUCT)&(df.grp=='iso')
keep=(df.grp=='legit')|y
trm=keep&mon.isin([7,8]); vam=keep&(mon==9)
d=lgb.Dataset(X[trm],y[trm].astype(int))
m=lgb.train(dict(objective='binary',learning_rate=0.05,num_leaves=31,min_data_in_leaf=200,feature_fraction=0.8,verbose=-1,scale_pos_weight=5),d,400)
p=m.predict(X[vam]); yy=y[vam].values
print('pooled iso-struct vs legit Sep AUC',round(fauc(yy,p),4))
lab=df.label[vam].values
for cls in STRUCT:
    mm=(lab==cls)|(lab=='legit'); print(cls,round(fauc(lab[mm]==cls,p[mm]),4))
imp=pd.Series(m.feature_importance('gain'),X.columns).sort_values(ascending=False); print((imp/imp.sum()).head(15).round(3))
# precision at top k
o=np.argsort(-p); 
for k in [100,500,2000]: print('prec@',k,yy[o[:k]].mean(),'base',yy.mean())
out=pd.DataFrame({'txn_id':df.txn_id[vam].values,'p_iso':p,'y':yy,'label':lab})
out.to_parquet('/tmp/hunt/gbm_iso_sep.parquet')
Xs=X[vam].copy(); Xs['p']=p; Xs['y']=yy; Xs['label']=lab; Xs['app_version']=df.app_version[vam].values
top=Xs.sort_values('p',ascending=False).head(300)
print(top.groupby('y')[['amount','sim_age_hours','device_age_days','own_dev','away','pin_retries','new_payee','login_to_txn_sec','amt_bal','n_receiver_id','s_hour_n']].median().T)
print(top.groupby('y').app_version.apply(lambda s:s.astype(str).str.startswith('3').mean()))
print(top.receiver_type.value_counts(), top.label.value_counts())
