"""Rule DEVBLOCK: Oct fraud-device ID block.
Device IDs are allocated sequentially by the generator: own phones DV0000000-0419999, then a FRAUD device block
(train: DV0420000-0428454, class-ordered sub-blocks), then legit second phones (train: 428455-473631, every ID used once).
October repeats the layout with new IDs: fraud block 473632..(start of contiguous run)-1, legit second phones after.
Rows in the Oct fraud block that the base submission calls legit -> class of the K nearest (by device number) base-predicted fraud rows."""
from base import *
import sys
def contiguous_start(ids, run=100):
    ids=np.sort(np.unique(ids)); g=np.diff(ids)
    for i in range(len(g)-run):
        if (g[i:i+run]==1).all(): return ids[i]
    return None
def knn_class(dn_q, dn_f, lab_f, K=5):
    out=[]
    for x in dn_q:
        idx=np.argsort(np.abs(dn_f-x),kind='stable')[:K]
        out.append(pd.Series(lab_f[idx]).value_counts().index[0])
    return np.array(out)
def apply(sub, te, tr_max_dev, K=5, verbose=True):
    t=te.set_index('txn_id').reindex(sub.id); dn=t.device_id.str[2:].astype(int).values
    new=dn>tr_max_dev
    start=dn[new].min(); end=contiguous_start(dn[new])
    blk=(dn>=start)&(dn<end)
    lab=sub.label.values.copy()
    f=blk&(lab!='legit'); q=blk&(lab=='legit')
    newlab=knn_class(dn[q],dn[f],lab[f],K)
    out=sub.copy(); out.loc[q,'label']=newlab
    if verbose:
        print('fraud block',start,end-1,'rows',blk.sum(),'base fraud',f.sum(),'flipped',q.sum()); print(pd.Series(newlab).value_counts().to_dict())
    return out, q
if __name__=='__main__':
    base=sys.argv[1] if len(sys.argv)>1 else R+'submissions/v9_v6_qrrule.csv'
    out_path=sys.argv[2] if len(sys.argv)>2 else '/tmp/hunt3/cand_v9_devblock.csv'
    te=pd.read_csv(R+'data/upi-fraud-rings/test.csv'); tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv',usecols=['device_id'])
    trmax=tr.device_id.str[2:].astype(int).max(); print('train max device',trmax)
    sub=pd.read_csv(base); out,q=apply(sub,te,trmax)
    assert len(out)==len(te) and out.id.is_unique
    out.to_csv(out_path,index=False); print('wrote',out_path,'changed',(out.label!=sub.label).sum(),'fraud share',round((out.label!='legit').mean(),4))
