# can txn_id be inverted as hash(counter)? -> generation order
from load import *
import hashlib
df=load(); ids=set(df.txn_id.str[2:].values)
print('n ids',len(ids),'sample',list(ids)[:3])
N=2_000_000
for name,f in [('md5',hashlib.md5),('sha1',hashlib.sha1),('sha256',hashlib.sha256)]:
    for fmt in ['{}','TX{}','txn{}','{:07d}','TX{:07d}','txn_{}','{:08d}']:
        hit=0
        for i in range(0,N,20):
            h=f(fmt.format(i).encode()).hexdigest()
            if h[:8] in ids or h[-8:] in ids: hit+=1
        print(name,fmt,'hits in 100k probes',hit)
