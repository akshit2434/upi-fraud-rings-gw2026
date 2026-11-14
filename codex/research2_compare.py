import sys,json
sys.path.insert(0,'src')
import numpy as np,pandas as pd
from common import xfit_f1,macro_f1,tune_weights
from features import LABELS
from sklearn.metrics import f1_score
Y=np.load('codex/research2_y.npy');ids=np.load('codex/research2_val_ids.npy',allow_pickle=True)
P=np.load('/tmp/V_lgb_s0.npy');assert np.array_equal(ids,np.load('/tmp/V_lgb_s0_ids.npy',allow_pickle=True))
for nm,q in [('baseline',P),('augmented',np.load('codex/research2_augmented_P.npy')),('augmented_unclean',np.load('codex/research2_augmented_unclean_P.npy'))]:
 xf=[xfit_f1(q,Y,seed=s) for s in [123,124,125]]
 w,s=tune_weights(q,Y,iters=2)
 print(json.dumps({'name':nm,'xfit':xf,'mean':np.mean(xf),'same_set':s,'same_set_fraud_share':float(((q*w).argmax(1)>0).mean()),'per_class_same_set':dict(zip(LABELS,f1_score(Y,(q*w).argmax(1),average=None)))}),flush=True)
