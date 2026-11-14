import pandas as pd, numpy as np, sys
sys.path.insert(0,'/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/research/hunt')
from load import load, STRUCT
df=load(); tr=df[df.is_test==0].copy()
tr['k']=np.where(tr.label=='legit','legit',np.where(tr.label.isin(STRUCT),'struct|','single|')+tr.grp)
tr['dn']=tr.device_id.str[2:].astype(int); tr['sn']=tr.sender_id.str[2:].astype(int)
tr['own']=tr.dn==tr.sn; tr['devacct']=np.where(tr.dn<420000,'AC'+tr.dn.astype(str).str.zfill(6),None)
fr=tr[tr.label!='legit']
# count of fraud rows each account participates in (as sender or P2P receiver)
cnt=pd.concat([fr.sender_id, fr.receiver_id[fr.receiver_type=='P2P']]).value_counts()
lcnt=pd.concat([tr.sender_id, tr.receiver_id[tr.receiver_type=='P2P']]).value_counts()
isf=tr.label!='legit'
def other(acc, self_contrib):
    c=acc.map(cnt).fillna(0)-self_contrib; return c>0
tr['s_inf']=other(tr.sender_id, isf.astype(int))
tr['r_inf']=np.where(tr.receiver_type=='P2P', other(tr.receiver_id, isf.astype(int)), np.nan)
tr['do_inf']=np.where(~tr.own & tr.devacct.notna(), tr.devacct.map(cnt).fillna(0)-(isf&(tr.devacct==tr.sender_id)).astype(int)>0, np.nan)
dcnt=fr.device_id.value_counts(); tr['d_inf']=tr.device_id.map(dcnt).fillna(0)-isf.astype(int)>0
# device owner relationship for other_acct devices: owner == receiver? owner pays/receives sender?
pairs=set(zip(tr.sender_id,tr.receiver_id))
m=~tr.own & tr.devacct.notna()
tr.loc[m,'own_is_recv']=(tr.devacct[m]==tr.receiver_id[m]).astype(float)
tr.loc[m,'own_pair']=[float(((a,b) in pairs) or ((b,a) in pairs)) for a,b in zip(tr.sender_id[m],tr.devacct[m])]
own_city=pd.read_csv('/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/data/upi-fraud-rings/accounts.csv').set_index('account_id')
tr.loc[m,'own_samecity']=(tr.devacct[m].map(own_city.home_city)==tr.sender_id[m].map(own_city.home_city)).astype(float)
tr.loc[m,'own_samebank']=(tr.devacct[m].map(own_city.bank_code)==tr.sender_id[m].map(own_city.bank_code)).astype(float)
pd.set_option('display.width',250)
print(tr.groupby('k')[['s_inf','r_inf','do_inf','d_inf','own_is_recv','own_pair','own_samecity','own_samebank']].mean().round(4).to_string())
print(tr.groupby('k')[['r_inf','do_inf','own_pair']].count().to_string())
tr[['txn_id','k','label','s_inf','r_inf','do_inf','own_pair','own_samecity']].to_parquet('/tmp/hunt4/ent.parquet')
