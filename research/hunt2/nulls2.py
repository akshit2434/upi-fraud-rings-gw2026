from load import *
pd.set_option('display.width',250); pd.set_option('display.max_columns',20)
tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv'); te=pd.read_csv(R+'data/upi-fraud-rings/test.csv')
sub=pd.read_csv(R+'submissions/v6_R1_bag.csv').rename(columns={'id':'txn_id','label':'v6'}); te=te.merge(sub,on='txn_id')
K=lambda x:x.txn_type+'|'+x.receiver_type+'|mis:'+x.qr_vpa_mismatch.fillna(-1).astype(int).astype(str)
tr['k']=K(tr); te['k']=K(te)
norm={'QR|MERCHANT|mis:0','PAY|MERCHANT|mis:-1','PAY|P2P|mis:-1','COLLECT|P2P|mis:-1','AUTOPAY|MERCHANT|mis:-1','REFUND|MERCHANT|mis:-1','QR|MERCHANT|mis:1'}
a=tr[~tr.k.isin(norm)]; b=te[~te.k.isin(norm)]
print('train label mix per anomalous pattern'); print(pd.crosstab(a.k,a.label).to_string())
print('test v6 per anomalous pattern'); print(pd.crosstab(b.k,b.v6).to_string())
print('anomalous totals: train per10k %.2f test per10k %.2f'%(len(a)/len(tr)*1e4,len(b)/len(te)*1e4))
print('train fraud rows by anomalous-flag share', (a.label!='legit').sum()/(tr.label!='legit').sum())
