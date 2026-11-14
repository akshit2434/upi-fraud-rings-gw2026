# per class: block IDs allocated (used+gaps) vs counts of rows/events of that class elsewhere
from load8 import *
df,r=base()
d1=pd.read_parquet(C+'blk_tr.parquet'); d2=pd.read_parquet(C+'blk_oct.parquet')
def alloc(dv):
    # smooth class: majority in window of 7 used devices
    order=['velocity_bot','mule_account','layering_chain','sim_swap','account_takeover','phishing_collect','qr_swap','refund_abuse','fake_cashback','loan_app_extortion','merchant_collusion']
    dv=dv.copy(); dv['cls2']=dv.cls
    # treat 'legit' devices as previous class
    dv.loc[dv.cls2=='legit','cls2']=np.nan; dv['cls2']=dv.cls2.ffill()
    dv['ids']=1+dv.gap_after.fillna(0)
    return dv.groupby('cls2').agg(used=('devn','size'),ids=('ids','sum'),rows=('n','sum'))
A=alloc(d1); B=alloc(d2)
tr=df[(df.is_test==0)&(df.label!='legit')]
tr=tr.assign(zone=np.where(tr.block,'block',np.where(tr.own,'own','other')))
z=pd.crosstab(tr.label,tr.zone+"_"+tr.g)
A=A.join(z)
A['gaps']=A.ids-A.used
print(A.to_string())
print(B.assign(gaps=B.ids-B.used).to_string())
