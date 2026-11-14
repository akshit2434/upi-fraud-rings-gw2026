from load import *
import itertools
pd.set_option('display.width',250); pd.set_option('display.max_rows',200); pd.set_option('display.max_colwidth',100)
tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv'); te=pd.read_csv(R+'data/upi-fraud-rings/test.csv')
acc=pd.read_csv(R+'data/upi-fraud-rings/accounts.csv'); mer=pd.read_csv(R+'data/upi-fraud-rings/merchants.csv')
print(acc.columns.tolist(), mer.columns.tolist())
sub=pd.read_csv(R+'submissions/v6_R1_bag.csv').rename(columns={'id':'txn_id','label':'v6'}); te=te.merge(sub,on='txn_id')
X=pd.concat([tr.assign(g=0),te.assign(g=1)],ignore_index=True)
a=acc.set_index('account_id') if 'account_id' in acc else acc.set_index(acc.columns[0])
X['home']=X.sender_id.map(a[[c for c in a.columns if 'city' in c][0]])
X['os']=X.sender_id.map(a[[c for c in a.columns if 'os' in c.lower()][0]])
X['kyc']=X.sender_id.map(a[[c for c in a.columns if 'kyc' in c.lower()][0]])
X['rkyc']=X.receiver_id.map(a[[c for c in a.columns if 'kyc' in c.lower()][0]]).fillna('MER')
m=mer.set_index(mer.columns[0]); X['mcat_reg']=X.receiver_id.map(m[[c for c in m.columns if 'cat' in c][0]]).fillna('P2P')
ts=pd.to_datetime(X.timestamp)
F=pd.DataFrame({'tt':X.txn_type,'rt':X.receiver_type,'cat':X.merchant_category.fillna('NA'),'mcat_eq':np.where(X.mcat_reg=='P2P','p2p',(X.mcat_reg==X.merchant_category).astype(str)),
 'qr':X.qr_vpa_mismatch.fillna(-1).astype(int).astype(str),'app':X.app,'pin':X.pin_retries.clip(0,3).astype(str),'newp':X.new_payee.astype(str),
 'own':(X.device_id.str[3:]==X.sender_id.str[2:]).astype(str),'away':(X.device_city!=X.home).astype(str),'os':X.os.astype(str),'kyc':X.kyc.astype(str),'rkyc':X.rkyc.astype(str),
 'hr':ts.dt.hour.astype(str),'cents0':((X.amount*100).round()%100==0).astype(str),
 'simz':(X.sim_age_hours<1).astype(str),'devz':(X.device_age_days<0.05).astype(str),'login0':(X.login_to_txn_sec<=2).astype(str)})
g=X.g.values; ntr=(g==0).sum(); nte=(g==1).sum(); fr=(X.label.notna()&(X.label!='legit')).values; v6=(X.v6.fillna('legit')!='legit').values
codes={c:pd.factorize(F[c])for c in F}
rows=[]
for k in (1,2,3):
  for combo in itertools.combinations(F.columns,k):
    key=np.zeros(len(F),np.int64); rad=[]
    for c in combo: key=key*len(codes[c][1])+codes[c][0]; rad.append(len(codes[c][1]))
    M=int(np.prod(rad)); a0=np.bincount(key[g==0],minlength=M); a1=np.bincount(key[g==1],minlength=M)
    f0=np.bincount(key[g==0],weights=fr[g==0],minlength=M); v1=np.bincount(key[g==1],weights=v6[g==1],minlength=M)
    exp1=a0*nte/ntr
    for kk in np.flatnonzero((a1>=8)&(a1>4*exp1+3)):
        lab=[];x=kk
        for c,r in zip(combo[::-1],rad[::-1]): lab.append(f'{c}={codes[c][1][x%r]}'); x//=r
        rows.append(('&'.join(lab[::-1]),k,a0[kk],a1[kk],exp1[kk],f0[kk]/max(a0[kk],1),v1[kk]/a1[kk]))
Rn=pd.DataFrame(rows,columns=['cell','k','train_n','test_n','test_expected','train_fraud','v6flag'])
Rn.to_parquet('novel.parquet')
Rn=Rn.sort_values(['k','test_n'],ascending=[True,False])
print(Rn.groupby('k').head(40).round(3).to_string())
