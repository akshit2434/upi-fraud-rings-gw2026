from load import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',100)
d=load(); d=d[~((d._is_test==0)&(~d._month.isin(['2026-08','2026-09'])))]
d['lab']=np.where(d.grp=='oct',d.v6,d.label)
x=d[(d.device_is_own==0)].copy()
x['eg']=x.merchant_category.isin(['ecommerce','gaming'])
for g in ['sep','oct']:
    s=x[x.grp==g]
    dev=s.groupby('device_id').agg(n=('txn_id','size'),ns=('sender_id','nunique'),eg=('eg','mean'),
        vel=('lab',lambda v:(v=='velocity_bot').mean()),fr=('lab',lambda v:(v!='legit').mean()),login=('login_to_txn_sec','median'))
    dev['spr']=dev.n/dev.ns
    dev['cls']=np.select([dev.n<2,(dev.ns>=3)&(dev.spr<2),(dev.spr>=2)],['single','many_senders','few_senders_many_rows'],'other')
    dev['egc']=pd.cut(dev.eg,[-.1,.5,1.01],labels=['eg<=.5','eg>.5'])
    dev['lg']=pd.cut(dev.login,[-1,10,1e9],labels=['login<=10','login>10'])
    agg=dev.groupby(['cls','egc','lg'],observed=True).apply(lambda q:pd.Series({'devices':len(q),'rows':q.n.sum(),'rows10k':q.n.sum()/(d.grp==g).sum()*1e4,
        'vel_share':(q.vel*q.n).sum()/q.n.sum(),'fraud_share':(q.fr*q.n).sum()/q.n.sum()}))
    print('==',g,'(labels' if g=='sep' else '(v6 preds',')'); print(agg.round(3).to_string())
