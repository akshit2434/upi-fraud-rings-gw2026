# 'other' zone: device number = another account's own phone. Is the owner linked to the sender / to fraud?
from load8 import *
df,r=base()
dn=df.devn
df['zone']=np.select([df.own, dn<420000],['own','other'],'x')
df['owner']='AC'+df.devn.fillna(0).astype(int).astype(str).str.zfill(6)
o=df[df.zone=='other'].copy()
tr=df[df.is_test==0]
fr=tr[tr.label!='legit']
fraud_acc=set(fr.sender_id)|set(fr.receiver_id[fr.receiver_id.str.startswith('AC')])
o['owner_fraud']=o.owner.isin(fraud_acc)
# pair relation: has the sender ever paid the owner or owner paid sender
p2p=df[df.receiver_type=='P2P']
pairs=set(zip(p2p.sender_id,p2p.receiver_id))
o['owner_pair']=[(s,w) in pairs or (w,s) in pairs for s,w in zip(o.sender_id,o.owner)]
o['owner_is_recv']=o.owner==o.receiver_id
g=o.groupby(['sender_id','device_id']).t.transform('size'); o['nsd']=g
o['ndev_senders']=o.groupby('device_id').sender_id.transform('nunique')
# owner's own phone used by owner?
own_users=set(df[df.own].sender_id)
o['owner_active']=o.owner.isin(own_users)
ot=o[o.is_test==0]
for c in ['owner_fraud','owner_pair','owner_is_recv','owner_active']:
    print(c); print(pd.crosstab(ot[c],ot.label=='legit',margins=True))
print(pd.crosstab(ot.nsd.clip(0,10),ot.label=='legit'))
print(pd.crosstab(ot.ndev_senders.clip(0,5),ot.label=='legit'))
print('test:',o[o.is_test==1][['owner_fraud','owner_pair','owner_is_recv','owner_active']].mean().round(3).to_dict())
print('train:',ot[['owner_fraud','owner_pair','owner_is_recv','owner_active']].mean().round(3).to_dict())
o[['txn_id','owner','owner_fraud','owner_pair','nsd','ndev_senders','owner_active']].to_parquet(C+'other8.parquet')
