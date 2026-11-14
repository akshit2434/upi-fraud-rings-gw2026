import pandas as pd, numpy as np, hashlib, sys
sys.path.insert(0,'/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/research/hunt')
from load import load
df=load()
ids=set(df.txn_id.str[2:])
print(df.txn_id.str.len().value_counts().head())
hits={}
for fn in ['md5','sha1','sha256']:
    for pat in ['{i}','TX{i}','txn{i}','{i:07d}','tx_{i}','T{i}']:
        c=0
        for i in range(0,20000):
            h=getattr(hashlib,fn)(pat.format(i=i).encode()).hexdigest()[:8]
            if h in ids: c+=1
        hits[(fn,pat)]=c
print({k:v for k,v in hits.items() if v>0} or 'no hash hits')
# expected random collisions: 20000*1.8e6/16^8
print('exp random', 20000*len(ids)/16**8)
