# gap structure inside the fraud device blocks
from load8 import *
df,r=base()
v=pd.read_csv(R+'submissions/v13_oofu.csv').rename(columns={'id':'txn_id','label':'pred'})
df=df.merge(v,on='txn_id',how='left')
df['lab']=np.where(df.is_test==1,df.pred,df.label)
def blk(lo,hi,nm,frauds_only=True):
    d=df[(df.devn>=lo)&(df.devn<=hi)]
    dv=d.groupby('devn').agg(n=('t','size'),cls=('lab',lambda s:s[s!='legit'].mode().iloc[0] if (s!='legit').any() else 'legit'),
        t0=('t','min'),t1=('t','max'),ns=('sender_id','nunique'),te=('is_test','mean')).reset_index()
    dv['gap_after']=dv.devn.shift(-1)-dv.devn-1
    # segment by class runs
    dv['run']=(dv.cls!=dv.cls.shift()).cumsum()
    seg=dv.groupby('run').agg(cls=('cls','first'),lo=('devn','min'),hi=('devn','max'),ndev=('devn','size'),rows=('n','sum'),gaps=('gap_after','sum'),t0=('t0','min'),t1=('t1','max'))
    seg['span']=seg.hi-seg.lo+1; seg['fill']=seg.ndev/seg.span
    seg['t0']=pd.Timestamp('2026-07-01')+pd.to_timedelta(seg.t0,'s'); seg['t1']=pd.Timestamp('2026-07-01')+pd.to_timedelta(seg.t1,'s')
    print('=====',nm,'devices',len(dv),'span',hi-lo+1)
    big=seg[seg.ndev>=5]
    pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
    print(big.to_string())
    print('gap length distribution',dv.gap_after.value_counts().sort_index().head(15).to_dict())
    # gap vs class
    print(dv.groupby('cls').agg(ndev=('devn','size'),gap_mean=('gap_after','mean'),rows=('n','sum')).round(2).to_string())
    # time order within class
    for c in dv.cls.unique():
        x=dv[dv.cls==c]
        if len(x)>20: print(c,'spearman devn~t0',round(x[['devn','t0']].corr('spearman').iloc[0,1],3))
    return dv
d1=blk(420000,428454,'train block'); d2=blk(473632,474551,'oct block')
d1.to_parquet(C+'blk_tr.parquet'); d2.to_parquet(C+'blk_oct.parquet')
