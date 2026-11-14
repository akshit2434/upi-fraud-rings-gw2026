from load import *
df=load()
df['cd']=df.device_age_days-df.t/86400; df['cs']=df.sim_age_hours-df.t/3600
df=df.sort_values(['sender_id','device_id','t'])
G=df.groupby(['sender_id','device_id'])
for c in ['cd','cs']:
    df['d_prev_'+c]=(df[c]-G[c].shift(1)); df['d_next_'+c]=(G[c].shift(-1)-df[c])
L=df[df.grp=='legit']
print(L[['d_prev_cd','d_prev_cs']].describe(percentiles=[.01,.1,.5,.9,.99]).round(3))
df['k']=np.where(df.grp=='legit','legit',np.where(df.grp=='test','test',df.label+'|'+df.grp))
for c in ['cd','cs']:
    tol=0.15 if c=='cd' else 0.2
    df['ok_'+c]=np.where(df['d_prev_'+c].isna(),np.nan,(df['d_prev_'+c].abs()<tol).astype(float))
    df['okn_'+c]=np.where(df['d_next_'+c].isna(),np.nan,(df['d_next_'+c].abs()<tol).astype(float))
print(df.groupby('k')[['ok_cd','okn_cd','ok_cs','okn_cs']].mean().round(3).to_string())
print(L.d_prev_cd.abs().pipe(lambda x: pd.cut(x,[0,0.01,0.05,0.1,0.2,0.5,1,5,1e9]).value_counts(normalize=True).sort_index()))
