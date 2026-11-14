from load import *
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
d=load(); d=d[d.grp.isin(['sep','oct'])].copy()
P=np.load('/tmp/P_bag4_R1.npy'); ids=np.load('/tmp/P_bag4_R1_ids.npy',allow_pickle=True); w=np.load(R+'preds/v6_R1_bag_weights.npy')
sp=pd.Series((P*w).argmax(1)!=0,index=ids)
d['sepflag']=d.txn_id.map(sp)
print('sep rows with model pred', d[d.grp=='sep'].sepflag.notna().mean(), 'sep flag rate', d[d.grp=='sep'].sepflag.mean())
keep=((d.grp=='sep')&(d.sepflag==False))|((d.grp=='oct')&(d.v6=='legit'))
x=d[keep].copy()
feats=['txn_type','receiver_type','merchant_category','qr_vpa_mismatch','hr','new_payee','pin_retries','login_to_txn_sec','sim_age_hours','device_age_days',
 'amt_over_bal','device_is_own','away_from_home','d_nacc_1h','d_nacc_p7d','d_n_5m','d_n_1h','s_n_10m','s_n_1h','s_n_1d','r_nsend_1h','r_nsend_1d','r_out_next1h',
 'r_next_out_dt','s_in_prev1h','s_last_in_dt','rev_dt','s_tiny_in_dt','s2_7d_ref','s2_7d_lend','s2_7d_late','p2_7d_n',
 'r_nround_senders_7d','r2_7d_round','s_ndev_p7d','amt_round100','amt_round1000','amt_le10','sender_kyc','recv_kyc','s_n_p7d','r_n_1d','p_n_p30d']
for c in ['r_nsend_1h','r_nsend_1d','r_n_1d','r2_7d_round','r_nround_senders_7d']: x.loc[x.receiver_type=='MERCHANT',c]=np.nan
for c in ['txn_type','receiver_type','merchant_category']: x[c]=x[c].astype('category')
X=x[feats]; y=(x.grp=='oct').astype(int).values
oof=np.zeros(len(x))
for tri,vai in StratifiedKFold(4,shuffle=True,random_state=0).split(X,y):
    m=lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=63,min_child_samples=50,verbose=-1,subsample=0.8,subsample_freq=1,colsample_bytree=0.8)
    m.fit(X.iloc[tri],y[tri]); oof[vai]=m.predict_proba(X.iloc[vai])[:,1]
print('adv AUC (model-legit rows only)', roc_auc_score(y,oof))
imp=pd.Series(m.booster_.feature_importance('gain'),index=feats).sort_values(ascending=False); print((imp/imp.sum()).head(15).round(3).to_dict())
x['adv']=oof; x[['txn_id','grp','adv']].to_parquet('adv2.parquet')
o=x[x.grp=='oct']; s=x[x.grp=='sep']
for q in [0.9,0.95,0.98,0.99]:
    th=np.quantile(oof,q); print(q,'oct rows above',(o.adv>th).sum(),'sep rows above',(s.adv>th).sum())
top=x[x.adv>np.quantile(oof,0.995)]
print('top0.5%: oct',(top.grp=='oct').sum(),'sep',(top.grp=='sep').sum())
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
print(top[top.grp=='oct'][['txn_type','receiver_type','merchant_category','qr_vpa_mismatch','hr','amount','new_payee','sim_age_hours','device_age_days','device_is_own','login_to_txn_sec','d_nacc_1h','s_n_1h','r_nsend_1h','adv']].sort_values('adv',ascending=False).head(30))
print(top[top.grp=='oct'].txn_type.value_counts().to_dict(), top[top.grp=='oct'].receiver_type.value_counts().to_dict(), top[top.grp=='oct'].qr_vpa_mismatch.isna().mean())
