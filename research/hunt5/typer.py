"""hunt5: dedicated fraud-TYPE model (fraud-labelled rows only), Jul-Aug -> Sep. Replaces v6's type on rows v6 flags.
usage: .venv/bin/python research/hunt5/typer.py [--relabel ''|data/relabel_R1.parquet] [--clean iso7|none|isoall]"""
import sys, argparse, numpy as np, pandas as pd, lightgbm as lgb
R='/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/'
sys.path.insert(0,R+'src'); sys.path.insert(0,'/tmp/hunt3')
import os; os.chdir(R)
from features import LABELS, load_cached, apply_drop
from common import get_conn, clean_bad, apply_relabel
from sklearn.metrics import f1_score
ap=argparse.ArgumentParser(); ap.add_argument('--relabel',default=''); ap.add_argument('--clean',default='iso7')
ap.add_argument('--iters',type=int,default=400); ap.add_argument('--bal',type=float,default=0.0); ap.add_argument('--tag',default='typer')
a=ap.parse_args()
X,meta=load_cached(0); X=apply_drop(X,'ip_rank_month'); ids=meta._txn_id.values
L2I={l:i for i,l in enumerate(LABELS)}; lab=meta._label.map(L2I).fillna(-1).astype(int).values
trn=meta._month.isin(['2026-07','2026-08']).values; sep=(meta._month=='2026-09').values
conn=get_conn(ids,until='2026-09-01'); bad=clean_bad(lab,conn,a.clean)
labt=apply_relabel(lab,ids,a.relabel,mask=trn) if a.relabel else lab
m=trn&~bad&(labt>0)
y=labt[m]-1
w=np.ones(len(y))
if a.bal>0:
    cnt=np.bincount(y,minlength=11); w=(cnt.mean()/cnt[y])**a.bal
p=dict(objective='multiclass',num_class=11,learning_rate=0.05,num_leaves=63,min_child_samples=20,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,verbose=-1,num_threads=10,seed=0)
mdl=lgb.train(p,lgb.Dataset(X[m],y,weight=w),a.iters)
Ps=mdl.predict(X[sep]); np.save(f'/tmp/hunt5/{a.tag}_sepP.npy',Ps); np.save(f'/tmp/hunt5/{a.tag}_sepids.npy',ids[sep])
from base import sep_pred
sp=sep_pred().set_index('txn_id').reindex(ids[sep])
typ=np.array(LABELS[1:])[Ps.argmax(1)]
labs=meta._label.values[sep]
fl=(sp.sp.values!='legit'); fr=fl&(labs!='legit')
print(a, 'n flagged fraud-labelled',fr.sum())
print('type acc  v6 %.4f   typer %.4f'%((sp.sp.values[fr]==labs[fr]).mean(),(typ[fr]==labs[fr]).mean()))
f=lambda yy,pp: f1_score(yy,pp,labels=LABELS,average='macro')
new=sp.sp.values.copy(); new[fl]=typ[fl]
print('Sep macro v6 %.4f  -> typer-typed %.4f'%(f(labs,sp.sp.values),f(labs,new)))
for blend in [0.3,0.5,0.7]:
    # blend: v6 class probs (renormalised over fraud) with typer probs
    P6=np.load('/tmp/P_bag4_R1.npy'); i6=np.load('/tmp/P_bag4_R1_ids.npy',allow_pickle=True); w6=np.load('preds/v6_R1_bag_weights.npy')
    P6=pd.DataFrame(P6*w6,index=i6).reindex(ids[sep]).values[:,1:]; P6=P6/P6.sum(1,keepdims=True)
    Pb=blend*Ps+(1-blend)*P6; tb=np.array(LABELS[1:])[Pb.argmax(1)]; nb=sp.sp.values.copy(); nb[fl]=tb[fl]
    print(' blend %.1f type acc %.4f macro %.4f'%(blend,(tb[fr]==labs[fr]).mean(),f(labs,nb)))
