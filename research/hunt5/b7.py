import pandas as pd, numpy as np, sys
sys.path.insert(0,'/tmp/hunt3'); from base import sep_pred
R='/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/'
df=pd.read_parquet('/tmp/hunt5/raw.parquet'); df['dn']=df.device_id.str[2:].astype(int)
df['blk']=((df.dn>=420000)&(df.dn<428455))|((df.dn>=473632)&(df.dn<474552))
sp=sep_pred(); v11=pd.read_csv(R+'submissions/v11_blockseg.csv').rename(columns={'id':'txn_id','label':'pred'})
sep=df[df.txn_id.isin(sp.txn_id)].merge(sp,on='txn_id').rename(columns={'sp':'pred'})
te=df[df.is_test==1].merge(v11,on='txn_id')
def links(x,name,lab=True):
    B=x[x.blk&(x.pred!='legit')]
    nb=x[~x.blk]
    for key,kk in [('receiver_id','receiver_id'),('sender_id','sender_id')]:
        m=nb[['txn_id',kk,'t','pred']+(['label'] if lab else [])].merge(B[[kk,'t','pred']].rename(columns={'t':'tb','pred':'pb'}),on=kk)
        if key=='receiver_id': m=m[m[kk].str.startswith('AC')]
        m['gap']=(m.t-m.tb).abs()/3600; m=m.sort_values('gap').drop_duplicates('txn_id')
        m['gb']=pd.cut(m.gap,[-1,1,6,24,168,1e9])
        f=lambda z:pd.Series({'n':len(z),'pred_legit':(z.pred=='legit').sum(),'fr_of_predlegit':((z.label!='legit')&(z.pred=='legit')).sum()/max((z.pred=='legit').sum(),1) if lab else np.nan,'same_as_pb_of_predlegit':((z.label==z.pb)&(z.pred=='legit')).sum()/max((z.pred=='legit').sum(),1) if lab else np.nan})
        print(name,key); print(m.groupby('gb').apply(f).round(3).to_string())
links(sep,'SEP51'); links(te,'OCT',lab=False)
