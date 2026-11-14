from load import *
df=load(); tr=df[df.is_test==0].copy(); tr['pos']=np.arange(len(tr))
tr['h']=tr.txn_id.str[2:].apply(lambda x:int(x,16))
leg=tr[tr.grp=='legit']
for cls in STRUCT+['phishing_collect','qr_swap']:
    for g in ['iso','conn']:
        fr=tr[(tr.label==cls)&(tr.grp==g)]; sub=pd.concat([fr,leg]); y=(sub.label==cls).values
        print(cls,g,round(fauc(y,sub.pos.values),3),round(fauc(y,sub.h.values),3), 'hex mod16 TV', round(0.5*(fr.h%16).value_counts(normalize=True).sub((leg.h%16).value_counts(normalize=True)).abs().sum(),3))
# autocorrelation of label in file order
f=(tr.label!='legit').values.astype(int); print('P(fraud next|fraud)',f[1:][f[:-1]==1].mean(),'base',f.mean())
# time order vs file order
print('corr pos-time', np.corrcoef(tr.pos,tr.t)[0,1])
print('dup txn ids', df.txn_id.duplicated().sum())
