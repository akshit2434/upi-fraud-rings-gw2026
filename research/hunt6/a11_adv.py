# adversarial: Sep51 model-legit rows vs Oct v12-legit rows on drift-safe raw features; look for coherent Oct-only groups
from load import *
import lightgbm as lgb
df=load()
acc=pd.read_csv(R+'data/upi-fraud-rings/accounts.csv').set_index('account_id')
m=sepval(df)
sub=pd.read_csv(R+'submissions/v12_tailfix.csv').rename(columns={'id':'txn_id','label':'pred'})
s=df.merge(m,on='txn_id',how='left').merge(sub,on='txn_id',how='left')
a=s[(s.pf<0.05)].copy(); b=s[(s.is_test==1)&(s.pred=='legit')].copy()
x=pd.concat([a,b]); y=(x.is_test==1).values.astype(int)
X=pd.DataFrame(index=x.index)
for c in ['receiver_type','txn_type','merchant_category','app']: X[c]=x[c].astype('category')
X['own']=x.own.astype(int); X['away']=(x.device_city!=x.sender_id.map(acc.home_city)).astype(int)
for c in ['sim_age_hours','device_age_days','new_payee','pin_retries','login_to_txn_sec','qr_vpa_mismatch']: X[c]=x[c]
X['qrnull']=x.qr_vpa_mismatch.isna().astype(int)
pass
X['hour']=x.ts.dt.hour
X['cents']=((x.amount*100).round()%100); X['round100']=(x.amount%100==0).astype(int)
X['devblk']=(x.devn>=420000).astype(int)
fold=np.random.RandomState(0).randint(0,2,len(x)); p=np.zeros(len(x))
for k in [0,1]:
    tr=fold!=k
    md=lgb.train(dict(objective='binary',learning_rate=0.05,num_leaves=31,min_data_in_leaf=100,verbose=-1),lgb.Dataset(X[tr],y[tr]),300)
    p[~tr]=md.predict(X[~tr])
print('adv AUC',fauc(y,p))
imp=pd.Series(md.feature_importance('gain'),X.columns).sort_values(ascending=False); print((imp/imp.sum()).head(10).round(3))
x['padv']=p
# odds: base ratio
r=y.mean(); x['lift']=(p/(1-p))/(r/(1-r))
t=x[x.is_test==1].sort_values('padv',ascending=False)
print('Oct rows with lift>5:',(t.lift>5).sum(),' >3:',(t.lift>3).sum(), 'Sep rows lift>5:',((x.is_test==0)&(x.lift>5)).sum())
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
print(t.head(40)[['receiver_type','txn_type','merchant_category','amount','balance_before','sim_age_hours','device_age_days','pin_retries','login_to_txn_sec','qr_vpa_mismatch','own','devn','lift']])
x[['txn_id','padv','lift']].to_parquet(C+'adv2.parquet')
