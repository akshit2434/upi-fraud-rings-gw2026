from load import *
df=load(); mer=pd.read_csv(R+'data/upi-fraud-rings/merchants.csv')
df=df.merge(mer,left_on='receiver_id',right_on='merchant_id',how='left')
df['onb']=(df.ts.dt.normalize()-pd.to_datetime(df.onboarded_on)).dt.days
df['r_pop']=df.groupby('receiver_id').txn_id.transform('size')
df['r_pop_month']=df.groupby(['receiver_id',df.ts.dt.month]).txn_id.transform('size')
df['cat_mis']=(df.category!=df.merchant_category)
M=df[df.receiver_type=='MERCHANT'].copy()
M['k']=np.where(M.grp.isin(['legit','test']),M.grp,np.where(M.label.isin(STRUCT),'struct|'+M.grp,'single|'+M.grp))
print(M.groupby('k').agg(n=('txn_id','size'),neg_onb=('onb',lambda x:(x<0).mean()),onb_med=('onb','median'),onb_lt90=('onb',lambda x:(x<90).mean()),pop_med=('r_pop','median'),catmis=('cat_mis','mean')).round(4))
leg=M[M.grp=='legit'].sample(200000,random_state=0)
for cls in STRUCT+['phishing_collect','qr_swap','refund_abuse','loan_app_extortion','fake_cashback']:
    fr=M[(M.label==cls)&(M.grp=='iso')]; sub=pd.concat([fr,leg]); y=(sub.label==cls).values
    print(cls,len(fr),'onb',round(fauc(y,sub.onb.values),3),'pop',round(fauc(y,sub.r_pop.values),3),'popm',round(fauc(y,sub.r_pop_month.values),3))
print(pd.to_datetime(mer.onboarded_on).describe())
