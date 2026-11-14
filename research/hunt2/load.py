import numpy as np, pandas as pd, os
R='/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/'
CACHE='/tmp/hunt2/frame.parquet'
CTX=['d_nacc_1h','d_nacc_p7d','d_n_5m','d_n_1h','s_n_10m','s_n_1h','s_n_1d','r_nsend_1h','r_nsend_1d','r_out_next1h','r_next_out_dt',
     's_in_prev1h','s_last_in_dt','amt_over_bal','device_is_own','away_from_home','recv_acc_age_days','merchant_age_days',
     'rev_dt','s_tiny_in_dt','s2_7d_ref','s2_7d_lend','s2_7d_late','p2_7d_n','r_nround_senders_7d','r2_7d_round','s_ndev_p7d',
     'amt_round100','amt_round1000','amt_just_under','amt_le10','amt_rank_month','sender_kyc','recv_kyc','s_n_p7d','r_n_1d','r_n_p30d','p_n_p30d']
def load():
    if os.path.exists(CACHE): return pd.read_parquet(CACHE)
    f=pd.read_parquet('/tmp/feat_v4_s0.parquet',columns=CTX+['_txn_id','_month','_is_test','_label'])
    f=f[f._month.isin(['2026-08','2026-09','2026-10','2026-11'])]
    tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv'); te=pd.read_csv(R+'data/upi-fraud-rings/test.csv')
    raw=pd.concat([tr,te.assign(label=np.nan)],ignore_index=True).drop(columns=['app_version'])
    d=f.merge(raw,left_on='_txn_id',right_on='txn_id',how='left')
    sub=pd.read_csv(R+'submissions/v6_R1_bag.csv').rename(columns={'id':'txn_id','label':'v6'})
    d=d.merge(sub,on='txn_id',how='left')
    P=np.load(R+'preds/v6_R1_bag_test_P.npy'); ids=pd.read_csv(R+'preds/v6_R1_bag_test_ids.csv').iloc[:,0].values
    pf=pd.DataFrame({'txn_id':ids,'pfraud':1-P[:,0]})
    d=d.merge(pf,on='txn_id',how='left')
    d['ts']=pd.to_datetime(d.timestamp); d['hr']=d.ts.dt.hour
    d['grp']=np.where(d._is_test==1,'oct',np.where(d._month=='2026-09','sep','aug'))
    d['fraud']=(d.label.notna()&(d.label!='legit')).astype(int)
    d.to_parquet(CACHE); return d
