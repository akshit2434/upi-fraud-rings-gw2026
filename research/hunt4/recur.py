import pandas as pd, numpy as np, sys
sys.path.insert(0,'/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/research/hunt')
from load import load, STRUCT
df=load(); df['mon']=df.timestamp.str[:7].where(df.is_test==0,'test')
df['k']=np.where(df.is_test==1,'test',np.where(df.label=='legit','legit',np.where(df.label.isin(STRUCT),'struct|','single|')+df.grp))
df['pair']=df.sender_id+'>'+df.receiver_id
pm=df.groupby('pair').mon.nunique(); df['pair_months']=df.pair.map(pm)
# same pair, same day-of-month (+-1) in another month (subscription-like)
df['dom']=df.ts.dt.day
x=df.groupby(['pair','dom']).mon.nunique(); df['pair_dom_months']=pd.Series(list(zip(df.pair,df.dom))).map(x).values
sm=df.groupby(['sender_id','merchant_category']).mon.nunique(); 
pd.set_option('display.width',250)
g=df.groupby(['k','txn_type']).agg(n=('pair_months','size'),recur=('pair_months',lambda s:(s>1).mean()),domrec=('pair_dom_months',lambda s:(s>1).mean()))
print(g.unstack('txn_type').round(3).to_string())
