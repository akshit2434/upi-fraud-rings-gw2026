import pandas as pd, numpy as np
P='/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/'
df=pd.read_parquet('/tmp/hunt5/raw.parquet')
df['mk']=(df.txn_type!='QR')&df.qr_vpa_mismatch.notna()
v10=pd.read_csv(P+'submissions/v10_v6_octrules.csv').rename(columns={'id':'txn_id','label':'pred'})
te=df[df.is_test==1].merge(v10,on='txn_id')
tr=df[df.is_test==0]
mkC=tr[tr.conn].groupby('label').mk.mean()
mkAll=tr[tr.label!='legit'].groupby('label').mk.mean()
o=pd.DataFrame({'oct_mk_by_pred':te[te.mk].groupby('pred').size(),'pred_v10':te.groupby('pred').size(),'mkC':mkC,'mk_all':mkAll})
o['gen_est']=o.oct_mk_by_pred/o.mkC; o['T_est']=o.oct_mk_by_pred/o.mk_all
print(o.round(3).to_string()); print(o.sum())
# flagged rows by marker status: share of v10 flags that are marker rows, per class, vs Sep same thing labelled
print((te[te.pred!='legit'].groupby('pred').mk.mean()).round(3))
