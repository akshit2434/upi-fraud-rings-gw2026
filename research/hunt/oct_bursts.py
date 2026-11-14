from load import *
df=load()
df['own']=(df.device_id.str[3:]==df.sender_id.str[2:])
sub=pd.read_csv(R+'submissions/v6_R1_bag.csv').rename(columns={'id':'txn_id','label':'v6'})
df=df.merge(sub,on='txn_id',how='left')
sep=df[(df.is_test==0)&(df.ts.dt.month==9)].sample(frac=0.51,random_state=0).copy()
octo=df[df.is_test==1].copy()
def burst(d,key,win):
    d=d.sort_values([key,'t']).copy(); out=np.zeros(len(d),int)
    k=d[key].values; t=d.t.values
    idx=np.r_[0,np.flatnonzero(k[1:]!=k[:-1])+1,len(k)]
    for a,b in zip(idx[:-1],idx[1:]):
        if b-a<2: continue
        tt=t[a:b]; out[a:b]=np.searchsorted(tt,tt+win,side='right')-np.searchsorted(tt,tt-win)-1
    d['nb']=out; return d
pd.set_option('display.width',250)
cols=['amount','balance_before','sim_age_hours','device_age_days','login_to_txn_sec','pin_retries','new_payee','own']
for key,win,filt in [('sender_id',600,lambda s:s.receiver_type=='P2P'),('receiver_id',3600,lambda s:s.receiver_type=='P2P')]:
    for name,d in [('sep51',sep),('oct',octo)]:
        s=burst(d,key,win); s=s[filt(s)&(s.nb>=2)]
        s['hr']=s.ts.dt.hour; s['amt_bal']=s.amount/s.balance_before
        print('==',key,name,len(s),'per10k %.2f'%(len(s)/len(d)*1e4))
        print(s[cols+['amt_bal','hr']].median().round(2).to_dict())
        print(' txn_type',s.txn_type.value_counts(normalize=True).round(2).to_dict())
        if name=='sep51': print(' labels',s.label.value_counts().head(6).to_dict())
        else: print(' v6 preds',s.v6.value_counts().head(6).to_dict())
