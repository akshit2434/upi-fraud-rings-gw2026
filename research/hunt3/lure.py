from base import *
tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv'); te=pd.read_csv(R+'data/upi-fraud-rings/test.csv')
v9=pd.read_csv(R+'submissions/v9_v6_qrrule.csv').rename(columns={'id':'txn_id','label':'v9'}); te=te.merge(v9,on='txn_id'); te['label']=te.v9
pd.set_option('display.width',250,'display.max_columns',30,'display.max_rows',300)
def pairs(x):
    x=x.copy(); x['t']=pd.to_datetime(x.timestamp)
    tiny=x[(x.amount<=10)&(x.receiver_type=='P2P')&(x.txn_type=='PAY')][['txn_id','sender_id','receiver_id','t','amount','label']]
    p=x[x.receiver_type=='P2P'][['txn_id','sender_id','receiver_id','t','amount','label','txn_type']]
    m=tiny.merge(p,left_on=['sender_id','receiver_id'],right_on=['receiver_id','sender_id'],suffixes=('','_b'))
    m=m[(m.t_b>m.t)&(m.t_b-m.t<pd.Timedelta('7D'))]; m['gap_h']=(m.t_b-m.t).dt.total_seconds()/3600
    a=m.amount_b; m['ju']=(a%1000==999)|(a%1000==990)|(a%500==499); m['pat']=np.where(m.ju,'ju',np.where(a%100==0,'r100',np.where(a.round(0)==a,'int','paise')))
    return m
m=pairs(tr); n=pairs(te)
m['ev']=m.label+'>'+m.label_b; n['ev']=n.label+'>'+n.label_b
k=m.ev.isin(['fake_cashback>fake_cashback','phishing_collect>phishing_collect'])
print(m[k].groupby('ev').agg(n=('gap_h','size'),gap=('gap_h','median'),amt=('amount_b','median'),lure=('amount','median'),collect=('txn_type',lambda s:(s=='COLLECT').mean()),ju=('ju','mean')).round(2))
print(pd.crosstab(m[k].ev,m[k].pat)); print(pd.crosstab(m[k].ev,pd.cut(m[k].gap_h,[0,1,6,24,72,168])))
print(n.groupby('ev').agg(n=('gap_h','size'),gap=('gap_h','median'),amt=('amount_b','median'),collect=('txn_type',lambda s:(s=='COLLECT').mean()),ju=('ju','mean')).round(2))
print(pd.crosstab(n.ev,n.pat))
print(pd.crosstab(m[k].ev,m[k].amount.round(0).clip(upper=10)))
print(pd.crosstab(m[k].ev,m[k].txn_type))
x=n[n.ev.isin(['fake_cashback>phishing_collect','phishing_collect>fake_cashback','fake_cashback>legit','legit>fake_cashback'])]
print(x[['txn_id','txn_id_b','ev','amount','amount_b','txn_type','gap_h']].to_string())
x.to_parquet('lure_incons.parquet')
