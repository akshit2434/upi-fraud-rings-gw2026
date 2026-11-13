import numpy as np
import common

def fast_f1(y,pred):
    counts=np.bincount(np.asarray(y,dtype=int)*12+pred,minlength=144).reshape(12,12)
    den=counts.sum(0)+counts.sum(1)
    return np.divide(2*np.diag(counts),den,out=np.zeros(12),where=den>0).mean()
def install():
    rng=np.random.default_rng(17)
    for _ in range(5):
        y=rng.integers(0,12,1000);p=rng.integers(0,12,1000)
        assert abs(common.macro_f1(y,p)-fast_f1(y,p))<1e-12
    common.macro_f1=fast_f1
