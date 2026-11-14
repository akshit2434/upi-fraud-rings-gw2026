import numpy as np,pandas as pd,json
from scipy.stats import binom
from scipy.optimize import minimize_scalar
train=np.concatenate([pd.read_csv('codex/research2_bursts_'+m+'.csv').n.values for m in ['Jul','Aug','Sep']]);train=train[train>=10]
obs=pd.read_csv('codex/research2_bursts_Oct.csv').n.values;obs=obs[obs>=10]
def loglik(p):
 probs=np.array([binom.pmf(k,train,p).mean()/binom.sf(9,train,p).mean() for k in obs]);return np.log(probs+1e-300).sum()
ps=np.linspace(.3,1,701);ll=np.array([loglik(p) for p in ps]);best=ps[ll.argmax()]
print(json.dumps({'full_training_events':len(train),'training_size_mean':float(train.mean()),'training_size_max':int(train.max()),'test_sizes':obs.tolist(),'test_mean':float(obs.mean()),'loglik_p51':float(loglik(.51)),'loglik_best':float(ll.max()),'best_p_grid':float(best),'relative_likelihood_p51':float(np.exp(loglik(.51)-ll.max())),'prob_atleast58_one_event_at_p51_n60':float(binom.sf(57,60,.51)),'caveat':'Assumes unchanged empirical pre-thinning event-size distribution; October event generation can drift.'},indent=2))
