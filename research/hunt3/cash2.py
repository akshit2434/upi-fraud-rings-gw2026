from base import *
tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv'); te=pd.read_csv(R+'data/upi-fraud-rings/test.csv')
v9=pd.read_csv(R+'submissions/v9_v6_qrrule.csv').rename(columns={'id':'txn_id','label':'v9'}); te=te.merge(v9,on='txn_id'); te['label']=te.v9
pd.set_option('display.width',250,'display.max_columns',30,'display.max_rows',200)
for name,x in [('train',tr),('test(v9)',te)]:
    x=x.copy(); x['t']=pd.to_datetime(x.timestamp)
    tiny=x[(x.amount<=10)&(x.receiver_type=='P2P')&(x.txn_type=='PAY')][['txn_id','sender_id','receiver_id','t','amount','label']]
    p=x[x.receiver_type=='P2P'][['txn_id','sender_id','receiver_id','t','amount','label','txn_type']]
    m=tiny.merge(p,left_on=['sender_id','receiver_id'],right_on=['receiver_id','sender_id'],suffixes=('','_b'))
    m=m[(m.t_b>m.t)&(m.t_b-m.t<pd.Timedelta('7D'))]
    print(name,'tiny',len(tiny),'with followup',m.txn_id.nunique())
    print(pd.crosstab(m.label,m.label_b))
    print(pd.crosstab(m.label_b,m.txn_type))
    # tiny with no follow-up
    nf=tiny[~tiny.txn_id.isin(m.txn_id)]; print('tiny no followup labels', nf.label.value_counts().to_dict())
    # big payments: followups. also for fraud big (cashback/phishing) without tiny in data
