import pandas as pd, numpy as np, sys
sys.path.insert(0,'/tmp/hunt3'); sys.path.insert(0,'/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/research/hunt')
from load import load
from base import sep_pred
df=load(); df['v3']=df.app_version.astype(str).str.startswith('3'); df['mk']=df.qr_vpa_mismatch.notna()&(df.txn_type!='QR')
s=df.merge(sep_pred(),on='txn_id')
# 'probably generated' = v3 or marker (non-QR)
s['pg']=s.v3|s.mk
ab={'legit':'L','mule_account':'mu','velocity_bot':'ve','layering_chain':'la','sim_swap':'si','account_takeover':'at','phishing_collect':'ph','qr_swap':'qr','refund_abuse':'re','fake_cashback':'fc','loan_app_extortion':'lo','merchant_collusion':'co'}
x=s[(s.label!='legit')&s.pg]
ct=pd.crosstab(x.label.map(ab),x.sp.map(ab)); print(ct.to_string())
print((ct.div(ct.sum(1),axis=0)).round(2)['L'].to_string())
# legit-labelled rows that are 'pg' and predicted fraud (unreported fraud?) 
L=s[s.label=='legit']; print('legit pg rows',L.pg.sum(),'pred fraud among them',(L[L.pg].sp!='legit').mean().round(3),'among non-pg',(L[~L.pg].sp!='legit').mean().round(4))
