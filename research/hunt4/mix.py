import pandas as pd, numpy as np, sys
sys.path.insert(0,'/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/research/hunt')
from load import load, STRUCT
df=load(); df['v3']=df.app_version.astype(str).str.startswith('3')
df['mk']=(df.txn_type!='QR')&df.qr_vpa_mismatch.notna()
tr=df[df.is_test==0]; te=df[df.is_test==1]
L=tr[tr.label=='legit']; v3L=L.v3.mean(); v3C=tr[tr.grp=='conn'].v3.mean()
print('v3 legit',round(v3L,3),'conn',round(v3C,3),'marker legit',round(L.mk.mean(),4),'conn',round(tr[tr.grp=='conn'].mk.mean(),3))
rows=[]
for c,g in tr[tr.label!='legit'].groupby('label'):
    iso=g[g.grp=='iso']; con=g[g.grp=='conn']
    # fraction of iso rows that are 'legit-generated' (a), using conn v3 rate of that class as fraud reference
    v3c=con.v3.mean(); a=(v3c-iso.v3.mean())/(v3c-v3L)
    rows.append(dict(cls=c,n=len(g),iso=len(iso),conn_v3=v3c,iso_v3=iso.v3.mean(),legitgen_frac_of_iso=a,inj=a*len(iso),
      mk_conn=con.mk.mean(),mk_iso=iso.mk.mean()))
R=pd.DataFrame(rows).set_index('cls').round(3); print(R.to_string())
inj=R.inj.sum(); n=len(tr); nf=(tr.label!='legit').sum()
print('train rows',n,'fraud',nf,'est legit-generated fraud',int(inj),'=',round(inj/n*100,2),'% of rows')
gen=tr[(tr.label!='legit')]; 
# marker rate among generated (non-injected) fraud: marked fraud / (fraud - injected)
mkf=gen.mk.sum(); print('marked fraud',mkf,'marker rate among generated fraud',round(mkf/(nf-inj),3), 'gen share',round((nf-inj)/n*100,2))
mt=te.mk.sum(); print('test marked',mt,'-> fraud marked ~',round(mt*0.88),'-> gen fraud in test ~',int(mt*0.88/(mkf/(nf-inj))),'=',round(mt*0.88/(mkf/(nf-inj))/len(te)*100,2),'%')
for m in ['2026-07','2026-08','2026-09']:
    x=tr[tr.timestamp.str[:7]==m]; print(m, 'fraud%',round((x.label!='legit').mean()*100,2),'marked/10k',round(x.mk.mean()*1e4,1),'iso-struct%',round(((x.grp=='iso')&x.label.isin(STRUCT)).mean()*100,2))
print('test marked/10k',round(te.mk.mean()*1e4,1))
