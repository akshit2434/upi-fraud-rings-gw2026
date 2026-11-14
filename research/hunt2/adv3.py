from load import *
d=load(); a=pd.read_parquet('adv2.parquet'); x=d.merge(a[['txn_id','adv']],on='txn_id')
th=np.quantile(x.adv,0.99); t=x[x.adv>th]
nul=(t.txn_type=='QR')&(t.receiver_type=='P2P')&t.qr_vpa_mismatch.isna()
print('top1% oct',(t.grp=='oct').sum(),'sep',(t.grp=='sep').sum(),'null-QR-P2P in top1%',nul.sum(),'of 140')
t=t[~nul]
k=t.txn_type+'|'+t.receiver_type+'|'+pd.cut(t.amt_over_bal,[-1,.3,.9,1.01,100]).astype(str)
c=pd.crosstab(k,t.grp); c['ex']=c.oct-c.sep; print(c.sort_values('ex').to_string())
print(t.groupby('grp')[['amt_over_bal','device_age_days','sim_age_hours','login_to_txn_sec','amount']].median())
