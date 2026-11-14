import pandas as pd, numpy as np
df=pd.read_parquet('/tmp/hunt5/raw.parquet'); df['dn']=df.device_id.str[2:].astype(int); df['sn']=df.sender_id.str[2:].astype(int)
df['mk']=(df.txn_type!='QR')&df.qr_vpa_mismatch.notna(); df['v3']=df.app_version.astype(str).str.startswith('3')
tr=df[df.is_test==0]; blk=(tr.dn>=420000)&(tr.dn<428455); own=tr.dn==tr.sn
L=tr[tr.label=='legit']; vL=L.v3.mean(); mL=L.mk.mean()
rows=[]
for c,x in tr[tr.label!='legit'].groupby('label'):
    b=x[blk.loc[x.index]]; vG=b.v3.mean(); mG=b.mk.mean()
    g_v3=(x.v3.mean()-vL)/(vG-vL); g_mk=(x.mk.mean()-mL)/(mG-mL)
    ob=x[own.loc[x.index]&x.conn]
    rows.append(dict(cls=c,nblk=len(b),v3_blk=vG,mk_blk=mG,v3_conn_own=ob.v3.mean(),mk_conn_own=ob.mk.mean(),v3_all=x.v3.mean(),mk_all=x.mk.mean(),gen_v3=g_v3,gen_mk=g_mk))
R=pd.DataFrame(rows).set_index('cls'); print(R.round(3).to_string()); print('legit v3',round(vL,3),'mk',round(mL,5))
for col in ['gen_v3','gen_mk']:
    g=R[col].clip(0,1); print(col,'ceiling macro (Sep-label world)',round((0.985+(2*g/(1+g)).sum())/12,4))
