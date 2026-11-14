from base import *
tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv',usecols=['txn_id','timestamp','receiver_id','receiver_type','sender_id','label'])
d=load(); s=pd.read_parquet('sepp.parquet')[['txn_id','sp']]; d=d.merge(s,on='txn_id',how='left')
pd.set_option('display.width',250,'display.max_columns',30,'display.max_rows',200)
def hist(h):
    h=h[(h.receiver_type=='P2P')&(h.label!='legit')]
    # dominant fraud class per receiver
    return h.groupby('receiver_id').label.agg(lambda s:s.value_counts().index[0]).rename('rh')
H_ja=hist(tr[tr.timestamp<'2026-09-01']); H_all=hist(tr[tr.timestamp<'2026-10-01'])
S=d[(d.grp=='sep')&(d.receiver_type=='P2P')].merge(H_ja,left_on='receiver_id',right_index=True,how='left')
O=d[(d.grp=='oct')&(d.receiver_type=='P2P')].merge(H_all,left_on='receiver_id',right_index=True,how='left')
S['rh']=S.rh.fillna('none'); O['rh']=O.rh.fillna('none')
S['tf']=S.label!='legit'; S['pf']=S.sp!='legit'; O['pf']=O.v9!='legit'
a=S.groupby('rh').agg(Sn=('tf','size'),Strue=('tf','mean'),Spred=('pf','mean'),
   Ssame=('tf',lambda s: np.nan))
for c in a.index:
    if c=='none': continue
    m=S.rh==c; a.loc[c,'Ssame']=(S.label[m]==c).mean(); a.loc[c,'Ssame_pred']=(S.sp[m]==c).mean()
b=O.groupby('rh').agg(On=('pf','size'),Opred=('pf','mean'))
for c in b.index:
    if c=='none': continue
    b.loc[c,'Osame_pred']=(O.v9[O.rh==c]==c).mean()
print(a.join(b).round(3))
# by txn_type for hist receivers
print('Sep hist receivers by type'); x=S[S.rh!='none']; print(x.groupby('txn_type').agg(n=('tf','size'),true=('tf','mean'),pred=('pf','mean')).round(3))
print('Oct hist receivers by type'); x=O[O.rh!='none']; print(x.groupby('txn_type').agg(n=('pf','size'),pred=('pf','mean')).round(3))
# Sep: hist receivers rows where true fraud but predicted legit, label dist
x=S[(S.rh!='none')&S.tf&~S.pf]; print('Sep missed on hist receivers',len(x)); print(pd.crosstab(x.rh,x.label))
S.to_parquet('S_recv.parquet'); O.to_parquet('O_recv.parquet')
