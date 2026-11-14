import sys; sys.path.insert(0,'/tmp/hunt5'); from load import *
df=load(); df.to_parquet('/tmp/hunt5/raw.parquet')
tr=df[df.is_test==0]
tr['grp']=np.where(tr.label=='legit','legit',np.where(tr.conn,'C_','I_')+tr.label.str[:6])
for c in ['receiver_type','txn_type']:
    print(pd.crosstab(tr.grp,tr[c],normalize='index').round(3))
print(tr.groupby('grp')[['amount','sim_age_hours','device_age_days','pin_retries','login_to_txn_sec','new_payee']].median())
print(tr.grp.value_counts())
