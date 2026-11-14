from base import *
d=load(); s=sep_pred(); print(len(s), s.sp.value_counts().head(3))
d=d.merge(s,on='txn_id',how='left'); d['pred']=np.where(d.grp=='oct',d.v9,d.sp)
d['anom']=np.where(d.txn_type=='QR', np.where(d.qr_vpa_mismatch.isna(),'QRblank',np.where(d.qr_vpa_mismatch==1,'QR1','QR0')), np.where(d.qr_vpa_mismatch.isna(),'ok','nonQRset'))
d['mk']=d.anom.isin(['nonQRset','QR1','QRblank'])
S=d[d.grp=='sep']; O=d[d.grp=='oct']; print(S.sp.isna().mean())
pd.set_option('display.width',250,'display.max_columns',20)
rows=[]
for c in LABELS[1:]:
    rows.append(dict(cls=c, sep_true_mk=((S.label==c)&S.mk).sum(), sep_true_ok=((S.label==c)&~S.mk).sum(),
      sep_pred_mk=((S.sp==c)&S.mk).sum(), sep_pred_ok=((S.sp==c)&~S.mk).sum(),
      oct_pred_mk=((O.v9==c)&O.mk).sum(), oct_pred_ok=((O.v9==c)&~O.mk).sum()))
t=pd.DataFrame(rows).set_index('cls')
t['sep_okmk_pred']=t.sep_pred_ok/t.sep_pred_mk; t['oct_okmk_pred']=t.oct_pred_ok/t.oct_pred_mk
t['oct_true_est']=(t.oct_pred_mk/t.sep_pred_mk*(t.sep_true_mk+t.sep_true_ok)).round()
print(t.round(2))
d[['txn_id','grp','sp','sp_pf','anom','mk']].to_parquet('/tmp/hunt3/sepp.parquet')
