from base import *
tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv',usecols=['txn_id','timestamp','sender_id','receiver_id','receiver_type','device_id','label'])
pd.set_option('display.width',250,'display.max_columns',30,'display.max_rows',200)
d=load(); s=pd.read_parquet('sepp.parquet')[['txn_id','sp']]; d=d.merge(s,on='txn_id',how='left'); d=d[d.grp!='aug']
d['pred']=np.where(d.grp=='oct',d.v9,d.sp)
def pool(h,key):
    f=h[h.label!='legit']; a=f.groupby(key).agg(nf=('label','size'),ncls=('label','nunique'))
    a['nall']=h.groupby(key).size().reindex(a.index); return a
for key in ['receiver_id','sender_id','device_id']:
    ja=pool(tr[tr.timestamp<'2026-09-01'],key); al=pool(tr[tr.timestamp<'2026-10-01'],key)
    S=d[(d.grp=='sep')].merge(ja,left_on=key,right_index=True,how='left'); O=d[d.grp=='oct'].merge(al,left_on=key,right_index=True,how='left')
    for X in (S,O): X['nfb']=pd.cut(X.nf.fillna(0),[-1,0,1,2,4,1e9],labels=['0','1','2','3-4','5+']).astype(str)
    S['tf']=S.label!='legit'
    a=S[S.sp=='legit'].groupby(['receiver_type','nfb']).agg(n=('tf','size'),tf=('tf','mean'))
    b=S.groupby(['receiver_type','nfb']).agg(nS=('tf','size'),tfS=('tf','mean'),predS=('sp',lambda s:(s!='legit').mean()))
    c=O.groupby(['receiver_type','nfb']).agg(nO=('pred','size'),predO=('pred',lambda s:(s!='legit').mean()),nO_leg=('pred',lambda s:(s=='legit').sum()))
    print('==',key,'(hist fraud count of entity) | Sep pred-legit rows true fraud rate'); print(b.join(a).join(c).round(3))
    S[S.sp=='legit'][S.nf.fillna(0)>=1].groupby('label').size().pipe(lambda z: print(z.to_dict()))
