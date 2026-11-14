from load import *
df=load()
tr=df[df.is_test==0].copy()
acc=pd.read_csv(R+'data/upi-fraud-rings/accounts.csv')
tr=tr.merge(acc,left_on='sender_id',right_on='account_id',how='left')
tr['lbb']=np.log1p(tr.balance_before); tr['lamt']=np.log1p(tr.amount)
tr=tr.sort_values(['sender_id','t']).reset_index(drop=True)
g=tr.groupby('sender_id')
n=g.t.transform('size'); tr['n']=n
feats={}
for c in ['lbb','lamt','login_to_txn_sec','sim_age_hours','device_age_days','sender_account_age_days','ip_risk_score']:
    s=g[c].transform('sum'); s2=g[c].transform(lambda x:(x**2).sum())
    loo=(s-tr[c])/(n-1); 
    feats['loo_'+c]=(tr[c]-loo).abs()
    feats['dprev_'+c]=(tr[c]-g[c].shift(1)).abs()
    feats['dnext_'+c]=(tr[c]-g[c].shift(-1)).abs()
# device / city / app consistency
for c in ['device_id','device_city','app']:
    mode=g[c].transform(lambda x:x.value_counts().index[0])
    cnt=tr.groupby(['sender_id',c])[c].transform('size')
    feats['share_'+c]=cnt/n
feats['city_home']=(tr.device_city!=tr.home_city).astype(float)
F=pd.DataFrame(feats); F['grp']=tr.grp; F['label']=tr.label; F['n']=n; F['month']=tr.ts.dt.month
F.to_parquet('/tmp/hunt/loo_feats.parquet')
F=F[F.n>=3]
rows=[]
for cls in STRUCT+['phishing_collect','qr_swap']:
    m=((F.label==cls)&(F.grp=='iso'))|(F.grp=='legit')
    sub=F[m]; y=(sub.label==cls).values
    r={'cls':cls,'n_iso':y.sum()}
    for c in feats: r[c]=round(auc(y,sub[c].values),3)
    rows.append(r)
pd.set_option('display.width',300)
print(pd.DataFrame(rows).set_index('cls').T.to_string())
