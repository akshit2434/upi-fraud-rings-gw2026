import pandas as pd, numpy as np, sys
sys.path.insert(0,'/tmp/hunt3'); from base import sep_pred
R='/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/'
df=pd.read_parquet('/tmp/hunt5/raw.parquet'); df['mk']=(df.txn_type!='QR')&df.qr_vpa_mismatch.notna()
s=df.merge(sep_pred(),on='txn_id')
v10=pd.read_csv(R+'submissions/v10_v6_octrules.csv').rename(columns={'id':'txn_id','label':'pred'})
te=df[df.is_test==1].merge(v10,on='txn_id')
o=pd.DataFrame({'sep_flag_mk':s[s.sp!='legit'].groupby('sp').mk.mean(),'sep_nflag':s.groupby('sp').size(),'oct_flag_mk':te[te.pred!='legit'].groupby('pred').mk.mean(),'oct_nflag':te.groupby('pred').size(),
 'sep_mk_rows':s[s.mk].groupby('sp').size(),'oct_mk_rows':te[te.mk].groupby('pred').size()})
print(o.round(3).to_string()); print('total marker rows sep',s.mk.sum(),'oct',te.mk.sum(),' flagged sep',(s.sp!='legit').sum(),'oct',(te.pred!='legit').sum())
