from base import *
tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv'); te=pd.read_csv(R+'data/upi-fraud-rings/test.csv')
acc=pd.read_csv(R+'data/upi-fraud-rings/accounts.csv'); mer=pd.read_csv(R+'data/upi-fraud-rings/merchants.csv')
v9=pd.read_csv(R+'submissions/v9_v6_qrrule.csv').rename(columns={'id':'txn_id','label':'v9'}); te=te.merge(v9,on='txn_id'); te['label']=te.v9
for n,x in [('train',tr),('test',te)]:
    x=x.copy()
    pre=x.receiver_id.str[:2]; bad=((x.receiver_type=='P2P')&(pre!='AC'))|((x.receiver_type=='MERCHANT')&(pre!='MR'))
    m=x.merge(mer,left_on='receiver_id',right_on='merchant_id',how='left')
    cm=(x.receiver_type=='MERCHANT').values&(m.category.values!=x.merchant_category.values)
    onb=(x.receiver_type=='MERCHANT').values&(pd.to_datetime(m.onboarded_on).values>pd.to_datetime(x.timestamp).values)
    a=x.merge(acc,left_on='sender_id',right_on='account_id',how='left')
    age=(pd.to_datetime(a.timestamp)-pd.to_datetime(a.opened_on)).dt.days
    agediff=(age-a.sender_account_age_days).abs()
    ra=x.merge(acc,left_on='receiver_id',right_on='account_id',how='left'); ropen=(x.receiver_type=='P2P').values&(pd.to_datetime(ra.opened_on).values>pd.to_datetime(x.timestamp).values)
    selfpay=x.sender_id==x.receiver_id
    devown=x.device_id.str[3:]==x.sender_id.str[2:]
    fr=x.label!='legit'
    for k,v in [('recv prefix bad',bad),('cat!=merchants.csv',cm),('merchant onboarded after txn',onb),('sender age diff>1d',agediff>1),('recv acct opened after txn',ropen),('self pay',selfpay)]:
        v=np.asarray(v); print(n,k,int(v.sum()),'fraud rate',round(fr[v].mean(),3) if v.sum() else '-')
    print(n,'agediff dist',agediff.describe().round(2).to_dict())
