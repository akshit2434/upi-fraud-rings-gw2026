import sys; sys.path.insert(0,'src')
import numpy as np
from common import xfit_f1
from smooth import smooth
y=np.load('/tmp/val_y_s0.npy'); c=np.load('/tmp/val_comp_s0.npy')
L=np.mean([np.load(f'/tmp/V_lgb_s{s}.npy') for s in range(4)],0)
C=np.mean([np.load(f'/tmp/V_cat_s{s}.npy') for s in range(2)],0)
A=np.load('/tmp/ag/P_val.npy'); V6=np.load('/tmp/P_bag4_R1.npy')
cands={'LC':.6*L+.4*C,'LCA':.4*L+.3*C+.3*A,'LCA_sm':smooth(.4*L+.3*C+.3*A,c,.5),'LCAeq_sm':smooth((L+C+A)/3,c,.5),'LA_sm':smooth(.5*L+.5*A,c,.5)}
for k,P in cands.items():
    v=np.array([xfit_f1(P,y,seed=s) for s in [1,2,3,4,5]]); print(k, v.mean().round(4), v.round(4), flush=True)
