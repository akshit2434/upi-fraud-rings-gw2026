import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
df=pd.read_parquet('/tmp/hunt5/raw.parquet')
outdeg=df.groupby('sender_id').size(); indeg=df[df.receiver_type=='P2P'].groupby('receiver_id').size()
indeg_s=df[df.receiver_type=='P2P'].groupby('receiver_id').sender_id.nunique()
p=df[(df.is_test==0)&(df.receiver_type=='P2P')].copy()
p['r_out']=p.receiver_id.map(outdeg).fillna(0); p['r_in']=p.receiver_id.map(indeg).fillna(0); p['s_in']=p.sender_id.map(indeg).fillna(0); p['s_out']=p.sender_id.map(outdeg).fillna(0)
p['grp']=np.where(p.label=='legit','legit',np.where(p.conn,'C','I'))
print(p.groupby('grp')[['r_out','r_in','s_in','s_out']].describe(percentiles=[.1,.5,.9]).T.round(1).to_string())
L=p[p.grp=='legit']
for lab in ['mule_account','velocity_bot','sim_swap','account_takeover','layering_chain','merchant_collusion']:
    I=p[(p.label==lab)&(~p.conn)]
    print(lab,len(I),{c:round(roc_auc_score(np.r_[np.ones(len(I)),np.zeros(len(L))],np.r_[I[c],L[c]]),3) for c in ['r_out','r_in','s_in','s_out']})
