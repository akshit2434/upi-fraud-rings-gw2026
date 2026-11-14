import pandas as pd, numpy as np
df=pd.read_parquet('/tmp/hunt5/raw.parquet')
df['tid']=df.txn_id.str[2:].apply(lambda h:int(h,16))
print(df.tid.describe()); print('unique',df.tid.is_unique)
d=df.sort_values('tid').reset_index(drop=True)
for k in ['sender_id','device_id','receiver_id']:
    print(k,'adjacent same share', (d[k].values[1:]==d[k].values[:-1]).mean())
print('corr tid vs t', np.corrcoef(d.tid,d.t)[0,1])
f=d.label.fillna('test').values; fr=(f!='legit')&(f!='test')
print('adjacent both fraud', (fr[1:]&fr[:-1]).mean(), 'expected', fr.mean()**2)
# low bits / modulo structure
for m in [2,3,7,16,256,1000]:
    tr=df[df.is_test==0]; g=tr.groupby(tr.tid%m).apply(lambda x:(x.label!='legit').mean())
    print(m, round(g.min(),4), round(g.max(),4))
