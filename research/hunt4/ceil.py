import pandas as pd, numpy as np, sys
sys.path.insert(0,'/tmp/hunt3'); sys.path.insert(0,'/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/research/hunt')
from load import load
df=load(); df['v3']=df.app_version.astype(str).str.startswith('3')
tr=df[df.is_test==0]; N=len(tr); NT=260582
v3L=tr[tr.label=='legit'].v3.mean()
sub=pd.read_csv('/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/submissions/v10_v6_octrules.csv').label.value_counts()
# Sep val v6 predictions
from base import sep_pred
sp=sep_pred(); sep=tr[tr.txn_id.isin(sp.txn_id)].merge(sp,on='txn_id')
rows=[]
for c,g in tr[tr.label!='legit'].groupby('label'):
    iso=g[g.grp=='iso']; con=g[g.grp=='conn']; a=np.clip((con.v3.mean()-iso.v3.mean())/(con.v3.mean()-v3L),0,1)
    gen=len(g)-a*len(iso); inj=a*len(iso)
    s=sep[sep.label==c]; rec_conn=(s[s.grp=='conn'].sp==c).mean(); rec_iso=(s[s.grp=='iso'].sp==c).mean()
    prec=(sep[sep.sp==c].label==c).mean()
    rows.append(dict(cls=c,gen_share=gen/len(g),T=len(g)/N*NT*1.115,gen=gen/N*NT*0.915,pred_v10=sub.get(c,0),sep_rec_conn=rec_conn,sep_rec_iso=rec_iso,sep_prec=prec,
                     sep_n=len(s),sep_npred=(sep.sp==c).sum()))
R=pd.DataFrame(rows).set_index('cls')
R.loc['velocity_bot','gen']*=0.5
R['ceilF1_gen']=2*R.gen/(R.gen+R['T'])
# implied current F1 assuming precision 0.85 on test
R['estF1_now']=2*0.85*R.pred_v10/(R.pred_v10+R['T'])
print(R.round(3).to_string())
lf=0.985
print('ceiling macro (perfect on generated, zero on injected):', round((lf+R.ceilF1_gen.sum())/12,4))
print('est current macro:', round((lf+R.estF1_now.sum())/12,4), 'sum T',int(R['T'].sum()),'sum gen',int(R.gen.sum()))
