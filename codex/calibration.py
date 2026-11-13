import os,sys,json
sys.path.insert(0,'src')
import numpy as np,pandas as pd
import common
from features import LABELS

def fastf(y,p):
    c=np.bincount(y*12+p,minlength=144).reshape(12,12)
    return np.divide(2*np.diag(c),c.sum(0)+c.sum(1),out=np.zeros(12),where=c.sum(0)+c.sum(1)>0).mean()
m=pd.read_parquet('/tmp/feat_s0.parquet');v=((m._is_test==0)&(m._month=='2026-09')).values
y=m._label[v].map(dict(zip(LABELS,range(12)))).values.astype(int)
# random split avoids conflating calibration with late-month drift; deterministic and disjoint.
a=np.random.default_rng(571).random(len(y))<.5
common.macro_f1=fastf
for name,path in [('baseline','/tmp/P_f3_noip_400.npy'),('bag4','/tmp/P_bag4_noip.npy'),('signature','codex/signature_P.npy'),('relabel','codex/relabel_P.npy'),('iso6','codex/iso6_P.npy'),('base_signature_blend','codex/base_signature_blend_P.npy'),('bag_signature_blend','codex/bag_signature_blend_P.npy')]:
    if not os.path.exists(path):continue
    P=np.load(path); assert len(P)==len(y)
    pred=np.empty(len(y),int)
    for mask in [a,~a]:
        w,_=common.tune_weights(P[mask],y[mask],iters=2);pred[~mask]=(P[~mask]*w).argmax(1)
    w,f=common.tune_weights(P,y,iters=2)
    print(name,'argmax',fastf(y,P.argmax(1)),'tuned same set',f,'crossfit',fastf(y,pred),flush=True)
