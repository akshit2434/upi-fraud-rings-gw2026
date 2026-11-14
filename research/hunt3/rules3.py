"""hunt3 rules applied on top of a base submission (default v9_v6_qrrule). Each rule changes only its targeted rows.
 A DEVBLOCK   : Oct fraud-device ID block (new IDs below the contiguous legit-second-phone run) -> base-legit rows get KNN(order-aware K=3 (train sub-block order), device number) class of base-fraud rows.
 A2 OLDBLOCK  : Oct rows on train fraud-block devices (DV0420000-0428454) called legit -> that device's majority train fraud label.
 A3 LEGITBLOCK: Oct rows on the contiguous legit second-phone run called fraud -> legit (train analogue 0.07% fraud).
 B QRM_BLANK  : QR -> MERCHANT, qr_vpa_mismatch blank, amount not a multiple of 100, base legit -> qr_swap.
 C LURE       : tiny (<=10) P2P PAY X->Y followed within 7d by P2P Y->X; events containing a fake_cashback row:
                base-legit members, and phishing follow-ups more than 6h after the lure -> fake_cashback.
 D QR_NOCAT   : QR -> P2P without merchant_category, base legit -> phishing_collect (train 79% phishing)."""
from base import *
from rule_devblock import contiguous_start, knn_class
from order_rule import order_class
import sys, argparse
def run(sub, te, tr, rules='ABCD', verbose=True):
    out=sub.copy(); t=te.set_index('txn_id').reindex(sub.id).reset_index().rename(columns={'id':'txn_id'})
    lab=out.label.values.copy(); log={}
    dn=t.device_id.str[2:].astype(int).values; trdn=tr.device_id.str[2:].astype(int)
    trmax=trdn.max()
    if 'A' in rules:
        new=dn>trmax; start=dn[new].min(); end=contiguous_start(dn[new])
        blk=(dn>=start)&(dn<end); f=blk&(lab!='legit'); q=blk&(lab=='legit')
        newc=order_class(dn[q],dn[f],out.label.values[f],3); lab[q]=newc; log['A_devblock']=int(q.sum()); log['A_classes']=pd.Series(newc).value_counts().to_dict()
        # A2 old block
        tb=tr[(trdn>=420000)&(trdn<428455)&(tr.label!='legit')]
        maj=tb.groupby('device_id').label.agg(lambda s:s.value_counts().index[0])
        q2=(dn>=420000)&(dn<428455)&(lab=='legit')&t.device_id.isin(maj.index).values
        lab[q2]=t.device_id[q2].map(maj).values; log['A2_oldblock']=int(q2.sum())
        q3=(dn>=end)&(lab!='legit'); lab[q3]='legit'; log['A3_legitblock']=int(q3.sum())
    if 'B' in rules:
        q=((t.txn_type=='QR')&(t.receiver_type=='MERCHANT')&t.qr_vpa_mismatch.isna()&(t.amount%100!=0)).values&(lab=='legit')
        lab[q]='qr_swap'; log['B_qrm_blank']=int(q.sum())
    if 'C' in rules:
        x=t[['txn_id','sender_id','receiver_id','receiver_type','txn_type','amount','timestamp']].copy(); x['lab']=lab; x['ts']=pd.to_datetime(x.timestamp); x['i']=np.arange(len(x))
        tiny=x[(x.amount<=10)&(x.receiver_type=='P2P')&(x.txn_type=='PAY')]; p=x[x.receiver_type=='P2P']
        m=tiny.merge(p,left_on=['sender_id','receiver_id'],right_on=['receiver_id','sender_id'],suffixes=('','_b'))
        m=m[(m.ts_b>m.ts)&(m.ts_b-m.ts<pd.Timedelta('7D'))]; m['gap']=(m.ts_b-m.ts).dt.total_seconds()/3600
        m['ev']=m.sender_id+'>'+m.receiver_id
        evc=m.groupby('ev').apply(lambda g: (g.lab=='fake_cashback').any() or (g.lab_b=='fake_cashback').any())
        m=m[m.ev.map(evc)]
        idx=set(m.i[m.lab=='legit'])|set(m.i_b[m.lab_b=='legit'])|set(m.i_b[(m.lab_b=='phishing_collect')&(m.gap>6)])
        idx=np.array(sorted(idx),dtype=int); lab[idx]='fake_cashback'; log['C_lure']=len(idx)
    if 'D' in rules:
        q=((t.txn_type=='QR')&(t.receiver_type=='P2P')&t.merchant_category.isna()).values&(lab=='legit'); lab[q]='phishing_collect'; log['D_qr_nocat']=int(q.sum())
    out['label']=lab
    if verbose: print(log, 'changed',int((out.label!=sub.label).sum()),'fraud share',round((out.label!='legit').mean(),4))
    return out
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--base',default=R+'submissions/v9_v6_qrrule.csv'); ap.add_argument('--rules',default='ABCD'); ap.add_argument('--out'); a=ap.parse_args()
    te=pd.read_csv(R+'data/upi-fraud-rings/test.csv'); tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv',usecols=['device_id','label'])
    sub=pd.read_csv(a.base); out=run(sub,te,tr,a.rules)
    assert len(out)==len(te) and out.id.is_unique and set(out.id)==set(te.txn_id) and set(out.label)<=set(LABELS)
    out.to_csv(a.out,index=False); print('wrote',a.out)
