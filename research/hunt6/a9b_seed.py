from load import *
import random
df=load(); ids=set(df.txn_id.str[2:].values)
def rate(vals): return sum(v in ids for v in vals)/len(vals)
seeds=[0,1,7,42,123,1234,12345,2024,2025,2026,31337,99,100,2023,8,13,17,21,3,5,11,10]
best=[]
for s in seeds:
    n=20000
    r=np.random.default_rng(s)
    a=[format(int(x),'08x') for x in r.integers(0,2**32,n)]; best.append(('dr_int32',s,rate(a)))
    r=np.random.default_rng(s); b=r.bytes(4*n); best.append(('dr_bytes',s,rate([b[4*i:4*i+4].hex() for i in range(n)])))
    np.random.seed(s); a=[format(int(x),'08x') for x in np.random.randint(0,2**32,n,dtype=np.int64)]; best.append(('np_legacy',s,rate(a)))
    random.seed(s); a=[format(random.getrandbits(32),'08x') for _ in range(n)]; best.append(('py_bits32',s,rate(a)))
    random.seed(s); a=['%032x'%random.getrandbits(128) for _ in range(n)]; best.append(('py_uuid128_head',s,rate([x[:8] for x in a]))); 
    random.seed(s); a=[format(random.randint(0,16**8-1),'08x') for _ in range(n)]; best.append(('py_randint',s,rate(a)))
    import uuid
    random.seed(s); a=[uuid.UUID(int=random.getrandbits(128),version=4).hex[:8] for _ in range(n)]; best.append(('py_uuid4',s,rate(a)))
    r=np.random.default_rng(s); a=[''.join(r.choice(list('0123456789abcdef'),8)) for _ in range(2000)]; best.append(('dr_choice',s,rate(a)))
b=sorted(best,key=lambda x:-x[2]); print(b[:10]); print('chance',len(ids)/2**32)
