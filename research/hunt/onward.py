from load import *
df=load()
P=df[df.receiver_type=='P2P'][['txn_id','receiver_id','t','amount','grp','label','is_test']]
S=df[['sender_id','t','amount']].rename(columns={'sender_id':'receiver_id','t':'ts2','amount':'a2'}).sort_values('ts2')
P=P.sort_values('t')
m=pd.merge_asof(P,S,left_on='t',right_on='ts2',by='receiver_id',direction='forward',allow_exact_matches=False)
m['dt_on']=(m.ts2-m.t)/60; m['ratio']=m.a2/m.amount
m['on60']=((m.dt_on<=60)).astype(float); m['on60_sim']=((m.dt_on<=60)&(m.ratio.between(0.7,1.0))).astype(float)
m['k']=np.where(m.grp.isin(['legit','test']),m.grp,m.label+'|'+m.grp)
print(m.groupby('k')[['on60','on60_sim']].mean().round(4).to_string())
