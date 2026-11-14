from load import *

df=load()
for k in ['sender_id','receiver_id','device_id']:
    df['n_'+k]=df.groupby(k).txn_id.transform('size')
    df['ntr_'+k]=df[df.is_test==0].groupby(k).txn_id.transform('size')
df['pair_n']=df.groupby(['sender_id','receiver_id']).txn_id.transform('size')
df['s_nrecv']=df.groupby('sender_id').receiver_id.transform('nunique')
df['r_nsend']=df.groupby('receiver_id').sender_id.transform('nunique')
df['s_p2p_share']=df.assign(p=(df.receiver_type=='P2P')).groupby('sender_id').p.transform('mean')
df['s_as_recv']=df.sender_id.map(df.receiver_id.value_counts()).fillna(0)
df['r_as_send']=df.receiver_id.map(df.sender_id.value_counts()).fillna(0)
tr=df[df.is_test==0]
leg=tr[tr.grp=='legit'].sample(300000,random_state=0)
feats=['n_sender_id','n_receiver_id','n_device_id','pair_n','s_nrecv','r_nsend','s_p2p_share','s_as_recv','r_as_send']
out={}
for cls in STRUCT+['phishing_collect','qr_swap','refund_abuse','loan_app_extortion','fake_cashback']:
    for g in ['iso','conn']:
        fr=tr[(tr.label==cls)&(tr.grp==g)]
        for p2p in [False,True]:
            a=fr[fr.receiver_type=='P2P'] if p2p else fr; b=leg[leg.receiver_type=='P2P'] if p2p else leg
            sub=pd.concat([a,b]); y=(sub.label==cls).values
            out[(cls,g,'p2p' if p2p else 'all')]={f:fauc(y,sub[f].values.astype(float)) for f in feats}
pd.set_option('display.width',250)
print(pd.DataFrame(out).T.round(3).to_string())
print(tr.groupby('grp')[feats].median())
