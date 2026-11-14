from load import *
df=load()
df['own']=(df.device_id.str[3:]==df.sender_id.str[2:])
sep=df[(df.is_test==0)&(df.ts.dt.month==9)].sample(frac=0.51,random_state=0)
octo=df[df.is_test==1]
def stats(d):
    r={}
    n=len(d)/1e4
    vb=(~d.own)&d.merchant_category.isin(['ecommerce','gaming'])
    r['nonown_ecom_gaming']=vb.sum()/n
    r['nonown_all']=(~d.own).sum()/n
    r['nonown_login<40']=((~d.own)&(d.login_to_txn_sec<40)).sum()/n
    r['QR->P2P']=((d.txn_type=='QR')&(d.receiver_type=='P2P')).sum()/n
    r['qr_mis1']=(d.qr_vpa_mismatch==1).sum()/n
    r['QR']=(d.txn_type=='QR').sum()/n
    r['COLLECT']=(d.txn_type=='COLLECT').sum()/n
    r['REFUND']=(d.txn_type=='REFUND').sum()/n
    r['sim<48h']=(d.sim_age_hours<48).sum()/n
    r['sim<48h_night']=((d.sim_age_hours<48)&(d.ts.dt.hour<6)).sum()/n
    r['pin>=2']=(d.pin_retries>=2).sum()/n
    r['amt>20k_P2P']=((d.amount>20000)&(d.receiver_type=='P2P')).sum()/n
    r['amt<=10_P2P']=((d.amount<=10)&(d.receiver_type=='P2P')).sum()/n
    r['lending']=(d.merchant_category=='lending').sum()/n
    r['amt_x999']=((d.amount%1000).round(2).isin([999,499,999.0])).sum()/n
    r['round1000_merchant']=((d.amount%1000==0)&(d.receiver_type=='MERCHANT')&(d.amount>=1000)).sum()/n
    # device with >=3 distinct senders in data
    ds=d.groupby('device_id').sender_id.transform('nunique'); r['dev_multi_sender>=3']=(ds>=3).sum()/n
    rs=d[d.receiver_type=='P2P'].groupby('receiver_id').sender_id.transform('nunique'); r['p2p_recv_nsend>=5']=(rs>=5).sum()/n
    return r
print(pd.DataFrame({'sep51':stats(sep),'oct':stats(octo)}).assign(ratio=lambda x:x.oct/x.sep51).round(2).to_string())
