from base import *
tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv'); te=pd.read_csv(R+'data/upi-fraud-rings/test.csv')
v9=pd.read_csv(R+'submissions/v9_v6_qrrule.csv').rename(columns={'id':'txn_id','label':'v9'}); te=te.merge(v9,on='txn_id')
def mk(x):
    mis=x.qr_vpa_mismatch
    x['anom']=np.where(x.txn_type=='QR', np.where(mis.isna(),'QRblank',np.where(mis==1,'QR1','QR0')), np.where(mis.isna(),'ok','nonQRset'))
    x['mon']=x.timestamp.str[:7]
for x in (tr,te): mk(x)
tr=tr[tr.mon<'2026-10']
pd.set_option('display.width',250,'display.max_columns',30)
for m in ['2026-07','2026-08','2026-09']:
    s=tr[tr.mon==m]; print(m); print(pd.crosstab(s.label,s.anom).T.round(0))
print('OCT v9'); print(pd.crosstab(te.v9,te.anom).T)
# per class: share of class rows with any marker (nonQRset or QR1 or QRblank)
tr['mark']=tr.anom.isin(['nonQRset','QR1','QRblank'])
print(tr.groupby(['label','mon']).mark.mean().unstack().round(3))
