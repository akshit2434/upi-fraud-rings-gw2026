import pandas as pd, numpy as np
R='/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/'
df=pd.read_parquet('/tmp/hunt5/raw.parquet'); df['dn']=df.device_id.str[2:].astype(int); df['sn']=df.sender_id.str[2:].astype(int)
v11=pd.read_csv(R+'submissions/v11_blockseg.csv').rename(columns={'id':'txn_id','label':'pred'})
te=df[df.is_test==1].merge(v11,on='txn_id'); tr=df[df.is_test==0]
for nm,x in [('train',tr),('test',te)]:
    no=x[x.dn!=x.sn]
    print(nm,'non-own rows',len(no),'dn ranges:')
    h=pd.cut(no.dn,[0,420000,428455,473632,474552,480000,500000,10**7])
    print(no.groupby(h).agg(rows=('txn_id','size'),devs=('dn','nunique'),mn=('dn','min'),mx=('dn','max')).to_string())
x=te[te.dn>=474552]; g=x.groupby('dn').size(); print('oct 2nd-run devices',len(g),'multi-row',(g>1).sum(),'max',g.index.max(),'flagged',(x.pred!='legit').sum())
x=tr[(tr.dn>=428455)&(tr.dn<473632)]; g=x.groupby('dn').size(); print('train 2nd-run devices',len(g),'multi',(g>1).sum(), 'ID span',473632-428455)
# non-own phones that are other accounts' phones in test, by v11 label
y=te[(te.dn!=te.sn)&(te.dn<420000)]; print('test other-account phones',len(y),y.pred.value_counts().head().to_dict())
