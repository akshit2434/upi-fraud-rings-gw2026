from base import *
d=load(); d=d[d.grp!='aug'].copy(); d['lab']=np.where(d.grp=='oct',d.v9,d.label)
pd.set_option('display.width',250,'display.max_columns',30,'display.max_rows',200)
x=d[(d.txn_type!='QR')&d.qr_vpa_mismatch.notna()&(d.receiver_type=='MERCHANT')]
print(pd.crosstab([x.cat],[x.grp,x.lab]).T)
