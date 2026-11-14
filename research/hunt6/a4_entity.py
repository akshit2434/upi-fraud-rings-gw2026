# entity-selection features: activity of sender / receiver / device; accounts.csv attrs of sender & receiver
from load import *
df=load()
acc=pd.read_csv(R+'data/upi-fraud-rings/accounts.csv'); mer=pd.read_csv(R+'data/upi-fraud-rings/merchants.csv')
F=pd.DataFrame({'txn_id':df.txn_id})
tr=df.is_test==0
# counts on train only (full density) to keep comparable
for k in ['sender_id','receiver_id','device_id']:
    vc=df[tr][k].value_counts(); F['n_'+k]=df[k].map(vc).fillna(0).values-tr.values
    vcm=df.groupby([k,'mon']).txn_id.transform('size'); F['nm_'+k]=vcm.values-1
F['s_as_recv']=df.sender_id.map(df[tr].receiver_id.value_counts()).fillna(0).values
F['r_as_send']=df.receiver_id.map(df[tr].sender_id.value_counts()).fillna(0).values
F['s_nrecv']=df.groupby('sender_id').receiver_id.transform('nunique').values
F['pair_n']=df.groupby(['sender_id','receiver_id']).txn_id.transform('size').values-1
F['s_nmon']=df.groupby('sender_id').mon.transform('nunique').values
F['s_ndev']=df.groupby('sender_id').device_id.transform('nunique').values
F['d_nsend']=df.groupby('device_id').sender_id.transform('nunique').values
a=acc.set_index('account_id')
for p,col in [('s','sender_id'),('r','receiver_id')]:
    F[p+'_open']=(df.ts-pd.to_datetime(df[col].map(a.opened_on))).dt.days.values
    F[p+'_kyc']=df[col].map(a.kyc_level).astype('category').cat.codes.values
    F[p+'_ios']=(df[col].map(a.os)=='ios').astype(float).where(df[col].map(a.os).notna()).values
    F[p+'_bank']=df[col].map(a.bank_code).str[1:].astype(float).values
    F[p+'_idn']=pd.to_numeric(df[col].str[2:],errors='coerce').values
F['r_home_eq_s']=(df.receiver_id.map(a.home_city)==df.sender_id.map(a.home_city)).astype(float).where(df.receiver_type=='P2P').values
F['dev_eq_home']=(df.device_city==df.sender_id.map(a.home_city)).astype(float).values
F['dev_eq_rhome']=(df.device_city==df.receiver_id.map(a.home_city)).astype(float).where(df.receiver_type=='P2P').values
mm=mer.set_index('merchant_id')
F['m_onb']=(df.ts-pd.to_datetime(df.receiver_id.map(mm.onboarded_on))).dt.days.values
F['m_city_eq']=(df.receiver_id.map(mm.city)==df.device_city).astype(float).where(df.receiver_type=='MERCHANT').values
F['m_city_home']=(df.receiver_id.map(mm.city)==df.sender_id.map(a.home_city)).astype(float).where(df.receiver_type=='MERCHANT').values
F['same_bank']=(df.receiver_id.map(a.bank_code)==df.sender_id.map(a.bank_code)).astype(float).where(df.receiver_type=='P2P').values
F.to_parquet(C+'ent_feats.parquet')
m=sepval(df); s=df[['txn_id','label','block','grp','mon','is_test','receiver_type']].merge(F,on='txn_id')
s=s.merge(m,on='txn_id',how='left')
lp=(s.label!='legit')&s.label.notna()&(~s.block)&(s.pf<0.05); ll=(s.label=='legit')&(s.pf<0.05)
cols=[c for c in F.columns if c!='txn_id']
x=s[lp|ll]; y=lp[lp|ll].values
res={'sep_LP':{c:fauc(y,x[c].astype(float)) for c in cols}}
# Jul-Aug stealth own-phone vs legit
t=s[(s.mon.isin([7,8]))&(s.grp.isin(['stealth','legit']))]
res['julaug_stealth']={c:fauc((t.grp=='stealth').values,t[c].astype(float)) for c in cols}
print(pd.DataFrame(res).round(3))
