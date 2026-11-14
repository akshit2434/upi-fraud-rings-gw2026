import pandas as pd, numpy as np
df=pd.read_parquet('/tmp/hunt5/raw.parquet')
tr=df[df.is_test==0].copy()
tr['mk']=(tr.txn_type!='QR')&tr.qr_vpa_mismatch.notna()
tr['v3']=tr.app_version.astype(str).str.startswith('3')
L=tr[tr.label=='legit']; print('legit mk',L.mk.mean().round(5),'v3',L.v3.mean().round(4))
rows=[]
for c,x in tr[tr.label!='legit'].groupby('label'):
    C=x[x.conn]; I=x[~x.conn]
    mC,mI=C.mk.mean(),I.mk.mean(); vC,vI=C.v3.mean(),I.v3.mean()
    # generated v3 rate among marker rows
    vM=x[x.mk].v3.mean()
    g_mk=(len(C)+len(I)*(mI-L.mk.mean())/(mC-L.mk.mean()))/len(x)
    g_v3=(x.v3.mean()-L.v3.mean())/(vM-L.v3.mean())
    rows.append(dict(cls=c,n=len(x),conn=len(C)/len(x),mkC=mC,mkI=mI,v3C=vC,v3I=vI,v3_markrows=vM,gen_mk=g_mk,gen_v3m=g_v3))
R=pd.DataFrame(rows).set_index('cls'); print(R.round(3).to_string())
for col in ['gen_mk','gen_v3m']:
    g=R[col].clip(0,1); print(col,'ceiling macro',round((0.985+(2*g/(1+g)).sum())/12,4))
