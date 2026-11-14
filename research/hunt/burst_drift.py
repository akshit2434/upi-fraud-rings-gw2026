from load import *
df=load()
df['own']=(df.device_id.str[3:]==df.sender_id.str[2:])
sep=df[(df.is_test==0)&(df.ts.dt.month==9)].sample(frac=0.51,random_state=0).copy()
octo=df[df.is_test==1].copy()
def burst(d,key,win=600):
    d=d.sort_values([key,'t']); out=np.zeros(len(d),int)
    k=d[key].values; t=d.t.values
    # count same-key rows within +-win via groupwise searchsorted
    idx=np.r_[0,np.flatnonzero(k[1:]!=k[:-1])+1,len(k)]
    for a,b in zip(idx[:-1],idx[1:]):
        if b-a<2: continue
        tt=t[a:b]; out[a:b]=np.searchsorted(tt,tt+win,side='right')-np.searchsorted(tt,tt-win)-1
    return out, d.index
for name,d in [('sep51',sep),('oct',octo)]:
    d=d.sort_index()
    for key,win in [('device_id',600),('sender_id',600),('receiver_id',3600)]:
        s=d.sort_values([key,'t']); v,_=burst(s,key,win); s['nb']=v
        if key=='receiver_id': s=s[s.receiver_type=='P2P']
        n=len(d)/1e4
        b=s[s.nb>=2]
        extra=''
        if name=='sep51': extra=' fraudshare=%.2f'%((b.label!='legit').mean())+' top labels '+str(b.label.value_counts().head(3).to_dict())
        print(name,key,win,'rows/10k with >=2 nbrs: %.2f'%(len(b)/n),'own %.2f'%b.own.mean(),'ecom/gaming %.2f'%b.merchant_category.isin(['ecommerce','gaming']).mean(),'P2P %.2f'%(b.receiver_type=='P2P').mean(),extra)
