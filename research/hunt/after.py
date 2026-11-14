from load import *
df=load().sort_values(['sender_id','t']).reset_index(drop=True)
g=df.groupby('sender_id')
df['gap_next']=(g.t.shift(-1)-df.t)/3600; df['gap_prev']=(df.t-g.t.shift(1))/3600
df['next_dev_change']=(g.device_id.shift(-1)!=df.device_id).astype(float).where(g.t.shift(-1).notna())
df['next_city_change']=(g.device_city.shift(-1)!=df.device_city).astype(float).where(g.t.shift(-1).notna())
df['is_last']=(g.cumcount(ascending=False)==0).astype(float)
df['rank_frac']=g.cumcount()/g.t.transform('size')
# receiver side: P2P receiver sends out after this row
first_send=df.groupby('sender_id').t.min()
tr=df[df.is_test==0]
leg=tr[tr.grp=='legit'].sample(300000,random_state=0)
fs=['gap_next','gap_prev','next_dev_change','next_city_change','is_last','rank_frac']
res={}
for cls in STRUCT+['phishing_collect','qr_swap','refund_abuse']:
    fr=tr[(tr.label==cls)&(tr.grp=='iso')]; sub=pd.concat([fr,leg]); y=(sub.label==cls).values
    res[cls]={f:round(fauc(y,sub[f].values),3) for f in fs}
print(pd.DataFrame(res).to_string())
