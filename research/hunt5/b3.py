import pandas as pd, numpy as np
R='/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/'
df=pd.read_parquet('/tmp/hunt5/raw.parquet'); df['dn']=df.device_id.str[2:].astype(int)
v11=pd.read_csv(R+'submissions/v11_blockseg.csv').rename(columns={'id':'txn_id','label':'v11'})
P=np.load(R+'preds/v6_R1_bag_test_P.npy'); w=np.load(R+'preds/v6_R1_bag_weights.npy'); ids=pd.read_csv(R+'preds/v6_R1_bag_test_ids.csv').iloc[:,0].values
LABS=['legit','mule_account','velocity_bot','layering_chain','sim_swap','account_takeover','phishing_collect','qr_swap','refund_abuse','fake_cashback','loan_app_extortion','merchant_collusion']
v6=pd.DataFrame({'txn_id':ids,'v6':np.array(LABS)[(P*w).argmax(1)],'pc':P[:,11]*w[11]/(P*w).sum(1)})
te=df[df.is_test==1].merge(v11,on='txn_id').merge(v6,on='txn_id')
t=te[(te.dn>=474440)&(te.dn<474560)].sort_values('dn')
pd.set_option('display.width',250)
print(t[['dn','v6','v11','txn_type','receiver_type','merchant_category','amount','qr_vpa_mismatch','pc']].to_string())
# train tail profile: collusion vs loan zone
tr=df[(df.is_test==0)&(df.dn>=428200)&(df.dn<428455)]
print(tr.groupby(tr.dn>=428302).apply(lambda x:pd.Series({'n':len(x),'int':(x.amount%1==0).mean(),'r100':(x.amount%100==0).mean(),'merch':(x.receiver_type=='MERCHANT').mean(),'qr':(x.txn_type=='QR').mean(),'lend':(x.merchant_category=='lending').mean(),'labs':x.label.value_counts().head(3).to_dict()})))
