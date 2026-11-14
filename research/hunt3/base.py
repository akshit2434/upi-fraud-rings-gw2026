import numpy as np, pandas as pd, os, sys
sys.path.insert(0,'/tmp/hunt2')
R='/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/'
LABELS=['legit','mule_account','velocity_bot','layering_chain','sim_swap','account_takeover','phishing_collect','qr_swap','refund_abuse','fake_cashback','loan_app_extortion','merchant_collusion']
C='/tmp/hunt3/frame.parquet'
def load():
    if os.path.exists(C): return pd.read_parquet(C)
    d=pd.read_parquet('/tmp/hunt2/frame.parquet')
    v9=pd.read_csv(R+'submissions/v9_v6_qrrule.csv').rename(columns={'id':'txn_id','label':'v9'})
    d=d.merge(v9,on='txn_id',how='left')
    d['cat']=d.merchant_category.fillna('NA'); d['mis']=d.qr_vpa_mismatch.map({0.0:'0',1.0:'1'}).fillna('NA')
    d.to_parquet(C); return d
def sep_pred():
    P=np.load('/tmp/P_bag4_R1.npy'); ids=np.load('/tmp/P_bag4_R1_ids.npy',allow_pickle=True)
    w=np.load(R+'preds/v6_R1_bag_weights.npy'); lab=np.array(LABELS)[(P*w).argmax(1)]
    return pd.DataFrame({'txn_id':ids,'sp':lab,'sp_pf':1-P[:,0]})
