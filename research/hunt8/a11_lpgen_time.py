# do LP counts of class c track GEN counts of class c over time (month/week/day)? => LP selected per event
from load8 import *
from scipy.stats import spearmanr, pearsonr
df,r=base()
tr=df[(df.is_test==0)&(df.ts<'2026-10-01')&(df.label!='legit')].copy()
tr['day']=tr.ts.dt.floor('D'); tr['wk']=tr.ts.dt.isocalendar().week.values
print(pd.crosstab([tr.label],[tr.mon,tr.g]).to_string())
for unit in ['wk','day']:
    rows=[]
    for c in sorted(tr.label.unique()):
        s=tr[tr.label==c]
        t=pd.crosstab(s[unit],s.g).reindex(columns=['GEN','LP'],fill_value=0)
        allidx=sorted(tr[unit].unique()); t=t.reindex(allidx,fill_value=0)
        rho=spearmanr(t.GEN,t.LP)[0]; pr=pearsonr(t.GEN,t.LP)[0]
        rows.append((c,int(t.GEN.sum()),int(t.LP.sum()),round(rho,3),round(pr,3),round(t.LP.std()/np.sqrt(t.LP.mean()),2)))
    print(unit); print(pd.DataFrame(rows,columns=['cls','GEN','LP','spearman','pearson','LP_disp']).to_string())
