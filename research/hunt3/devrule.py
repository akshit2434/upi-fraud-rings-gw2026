from base import *
pd.set_option('display.width',250,'display.max_columns',30,'display.max_rows',400)
d=load(); s=pd.read_parquet('sepp.parquet')[['txn_id','sp']]; d=d.merge(s,on='txn_id',how='left')
d['dn']=d.device_id.str[2:].astype(int)
def nb_major(x, W):
    # x: rows in block, with 'pred'; for each row, majority of non-legit preds among rows with |dn - dn_i|<=W (excluding itself)
    x=x.sort_values('dn').reset_index(drop=True); dn=x.dn.values; pr=x.pred.values; out=[]
    for i in range(len(x)):
        lo=np.searchsorted(dn,dn[i]-W,'left'); hi=np.searchsorted(dn,dn[i]+W,'right')
        lab=[pr[j] for j in range(lo,hi) if j!=i and pr[j]!='legit']
        out.append(pd.Series(lab).value_counts().index[0] if lab else 'legit')
    x['nb']=out; return x
# Sep validation
S=d[(d.grp=='sep')&(d.dn>=420000)&(d.dn<428400)].copy(); S['pred']=S.sp
for W in [10,25,50,100]:
    x=nb_major(S,W); L=x[x.pred=='legit']
    print('Sep W',W,'block rows',len(x),'v6-legit',len(L),'true fraud',round((L.label!='legit').mean(),3),'nb class == true label (among true fraud)',round((L.nb[L.label!='legit']==L.label[L.label!='legit']).mean(),3),
          '| nb==true overall',round((L.nb==L.label).mean(),3))
x=nb_major(S,25); L=x[x.pred=='legit']; print(pd.crosstab(L.label,L.nb))
# also how good is v6 on block fraud rows
print('Sep block v6 pred==label among true fraud', round((S.sp[S.label!='legit']==S.label[S.label!='legit']).mean(),3), 'v6 legit share', round((S.sp=='legit').mean(),3))
