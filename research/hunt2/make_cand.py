# Candidate: v6_R1_bag + rule "QR payment to a P2P receiver with blank qr_vpa_mismatch -> qr_swap" (pattern absent in all of train)
from load import *
te=pd.read_csv(R+'data/upi-fraud-rings/test.csv'); sub=pd.read_csv(R+'submissions/v6_R1_bag.csv')
m=(te.txn_type=='QR')&(te.receiver_type=='P2P')&te.qr_vpa_mismatch.isna()
ids=set(te.txn_id[m]); s=sub.copy(); hit=s.id.isin(ids)
print('rows changed',hit.sum(),'previous labels',s.label[hit].value_counts().to_dict())
s.loc[hit,'label']='qr_swap'; assert len(s)==len(te) and s.id.is_unique
s.to_csv('/tmp/hunt2/cand_v6_qrnull.csv',index=False); print(s.label.value_counts().to_dict())
