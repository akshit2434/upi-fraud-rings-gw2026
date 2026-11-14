from base import *
from order_rule import order_class
d=load(); s=pd.read_parquet('sepp.parquet')[['txn_id','sp']]; d=d.merge(s,on='txn_id',how='left'); d['dn']=d.device_id.str[2:].astype(int)
S=d[(d.grp=='sep')&(d.dn>=420000)&(d.dn<428455)&(d.sp!='legit')].reset_index(drop=True)
dn=S.dn.values; lab=S.sp.values; y=S.label.values; tf=y!='legit'
# leave-one-out neighbour class (exclude same device rows to avoid trivially copying own device's prediction)
nb=[]
for i in range(len(S)):
    m=S.device_id.values!=S.device_id.values[i]
    nb.append(order_class(np.array([dn[i]]),dn[m],lab[m],3)[0])
nb=np.array(nb)
print('v6 class acc',round((lab[tf]==y[tf]).mean(),3),'neighbour class acc',round((nb[tf]==y[tf]).mean(),3))
dis=nb!=lab; print('disagree',dis.sum(),'v6 right',(lab[dis&tf]==y[dis&tf]).sum(),'nb right',(nb[dis&tf]==y[dis&tf]).sum())
print(pd.crosstab(pd.Series(lab[dis],name='v6'),pd.Series(nb[dis],name='nb')))
print(pd.crosstab(pd.Series(y[dis],name='true'),pd.Series(nb[dis],name='nb')))
