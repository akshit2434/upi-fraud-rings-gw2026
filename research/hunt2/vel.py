from load import *
pd.set_option('display.width',250); pd.set_option('display.max_columns',20)
d=load(); d=d[~((d._is_test==0)&(~d._month.isin(['2026-08','2026-09'])))]
d['night']=d.hr<6
d['lab']=np.where(d.grp=='oct',d.v6,d.label)
for nm,m in [('sburst s_n_10m>=2',d.s_n_10m>=2),('devmulti d_nacc_1h>=2',d.d_nacc_1h>=2),('dev burst d_n_5m>=3',d.d_n_5m>=3)]:
    x=d[m]
    print('==',nm); print(pd.crosstab([x.grp,x.night],x.lab).to_string())
    print(x.groupby('grp').apply(lambda s:pd.Series({'n':len(s),'own':s.device_is_own.mean(),'ecom_gam':s.merchant_category.isin(['ecommerce','gaming']).mean(),'p2p':(s.receiver_type=='P2P').mean(),'amt_med':s.amount.median(),'login_med':s.login_to_txn_sec.median(),'night':s.night.mean()})).round(3))
