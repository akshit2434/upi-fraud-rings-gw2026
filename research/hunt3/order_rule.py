from base import *
ORDER=['velocity_bot','mule_account','velocity_bot','layering_chain','sim_swap','account_takeover','phishing_collect','qr_swap','refund_abuse','fake_cashback','loan_app_extortion','merchant_collusion']
def order_class(dn_q, dn_f, lab_f, K=5):
    o=np.argsort(dn_f,kind='stable'); dn_f=dn_f[o]; lab_f=lab_f[o]; out=[]
    for x in dn_q:
        i=np.searchsorted(dn_f,x)
        L=lab_f[max(i-K,0):i]; Rr=lab_f[i:i+K]
        Lc=pd.Series(L).value_counts().index[0] if len(L) else None; Rc=pd.Series(Rr).value_counts().index[0] if len(Rr) else None
        if Lc is None: out.append(Rc); continue
        if Rc is None or Lc==Rc: out.append(Lc); continue
        # positions in train order: find a Lc position followed (later) by an Rc position with something in between
        best=None
        for a in [j for j,c in enumerate(ORDER) if c==Lc]:
            for b in [j for j,c in enumerate(ORDER) if c==Rc and j>a]:
                if best is None or b-a<best[1]-best[0]: best=(a,b)
        if best and best[1]-best[0]==2: out.append(ORDER[best[0]+1]); continue
        # else nearest side
        dl=x-dn_f[i-1] if i>0 else np.inf; dr=dn_f[i]-x if i<len(dn_f) else np.inf
        out.append(Lc if dl<=dr else Rc)
    return np.array(out)
if __name__=='__main__':
    from rule_devblock import knn_class
    d=load(); s=pd.read_parquet('sepp.parquet')[['txn_id','sp']]; d=d.merge(s,on='txn_id',how='left'); d['dn']=d.device_id.str[2:].astype(int)
    S=d[(d.grp=='sep')&(d.dn>=420000)&(d.dn<428455)]
    f=S.sp!='legit'; q=~f
    for K in [3,5]:
        a=knn_class(S.dn.values[q],S.dn.values[f],S.sp.values[f],K); b=order_class(S.dn.values[q],S.dn.values[f],S.sp.values[f],K)
        y=S.label.values[q]; tf=y!='legit'
        print('K',K,'knn acc',round((a[tf]==y[tf]).mean(),3),'order acc',round((b[tf]==y[tf]).mean(),3),'n',q.sum())
    print(pd.crosstab(y,b))
