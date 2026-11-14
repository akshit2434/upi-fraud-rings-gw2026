# second-phone (non-own, non-block) devices: ID allocation vs label / time
from load import *
df=load()
x=df[(~df.own)&(~df.block)].copy()
print('non-own nonblock rows',len(x), x.groupby('is_test').size().to_dict())
print(x.groupby('grp').size())
x['fr']=(x.label.notna()&(x.label!='legit')).astype(float); x.loc[x.is_test==1,'fr']=np.nan
# rows per device
x['nd']=x.groupby('device_id').txn_id.transform('size')
print(x.groupby('grp').nd.describe())
# deciles of devn within each period by alloc
x=x.sort_values('devn')
x['bin']=(x.devn//5000).astype(int)
g=x.groupby('bin').agg(n=('fr','size'),fr=('fr','mean'),test=('is_test','mean'),tmin=('ts','min'),tmax=('ts','max'),nsend=('sender_id','nunique'))
pd.set_option('display.width',200); pd.set_option('display.max_rows',300)
print(g)
