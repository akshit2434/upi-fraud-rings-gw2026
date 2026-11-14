from load import *
import itertools, sys
d=load(); d=d[~((d._is_test==0)&(~d._month.isin(['2026-08','2026-09'])))].copy()
cat=np.where(d.receiver_type=='P2P','p2p',d.merchant_category.fillna('na'))
a=d.amount.values
amtp=np.select([a<=10,d.amt_just_under>0,(a%1000==0),(a%100==0)],['le10','justunder','r1000','r100'],'other')
mag=pd.cut(a,[-1,100,1000,5000,20000,1e12],labels=['<100','<1k','<5k','<20k','20k+']).astype(str)
hb=pd.cut(d.hr,[-1,5,11,17,20,23],labels=['night','morn','day','eve','late']).astype(str)
F=pd.DataFrame({
 'tt':d.txn_type.values,'cat':cat,'amtp':amtp,'mag':mag,'hb':hb,'newp':d.new_payee.values,
 'simfresh':(d.sim_age_hours<72).astype(int).values,'notown':(1-d.device_is_own).astype(int).values,
 'devnew':(d.device_age_days<2).astype(int).values,'away':d.away_from_home.astype(int).values,
 'pin':(d.pin_retries>0).astype(int).values,'fastlogin':(d.login_to_txn_sec<40).astype(int).values,
 'drain':(d.amt_over_bal>=0.9).astype(int).values,'qrmis':(d.qr_vpa_mismatch==1).astype(int).values,
 'devmulti':(d.d_nacc_1h>=2).astype(int).values,'sburst':(d.s_n_10m>=2).astype(int).values,
 'rfanin':((d.receiver_type=='P2P')&(d.r_nsend_1h>=3)).astype(int).values,
 'onward':(d.r_out_next1h>0).astype(int).values,'inprev':(d.s_in_prev1h>0).astype(int).values,
 'newacc':(d.sender_account_age_days<30).astype(int).values,
 'newrecv':(d.recv_acc_age_days<30).astype(int).values,
},index=d.index)
F.to_parquet('/tmp/hunt2/F.parquet')
g=d.grp.values; N={k:(g==k).sum() for k in ['aug','sep','oct']}
gi=np.select([g=='aug',g=='sep'],[0,1],2)
fraud=d.fraud.values.astype(float); v6f=(d.v6.fillna('legit')!='legit').values.astype(float)
codes={}; levels={}
for c in F.columns:
    cc,u=pd.factorize(F[c].astype(str)); codes[c]=cc; levels[c]=u
cols=list(F.columns); rows=[]
for k in (1,2,3):
  for combo in itertools.combinations(cols,k):
    key=np.zeros(len(F),np.int64); rad=[]
    for c in combo: key=key*len(levels[c])+codes[c]; rad.append(len(levels[c]))
    M=int(np.prod(rad))
    cnt=[np.bincount(key[gi==j],minlength=M) for j in range(3)]
    trm=gi<2
    frs=np.bincount(key[trm],weights=fraud[trm],minlength=M)
    sfr=np.bincount(key[gi==1],weights=fraud[gi==1],minlength=M)
    v6=np.bincount(key[gi==2],weights=v6f[gi==2],minlength=M)
    for kk in np.flatnonzero(cnt[2]>=25):
        lab=[]; x=kk
        for c,r in zip(combo[::-1],rad[::-1]): lab.append(c+'='+levels[c][x%r]); x//=r
        n_tr=cnt[0][kk]+cnt[1][kk]
        rows.append(('&'.join(combo),'&'.join(lab[::-1]),cnt[0][kk],cnt[1][kk],cnt[2][kk],frs[kk]/max(n_tr,1),sfr[kk],v6[kk]/cnt[2][kk]))
R_=pd.DataFrame(rows,columns=['fields','cell','aug','sep','oct','frshare_trn','sep_fraud_n','v6flag_oct'])
for x in ['aug','sep','oct']: R_[x+'_10k']=R_[x]/N[x]*1e4
R_['excess_10k']=R_.oct_10k-R_.sep_10k
R_['excess_rows']=R_.oct-R_.sep*N['oct']/N['sep']
R_['ratio']=(R_.oct+0.5)/(R_.sep*N['oct']/N['sep']+0.5)
R_['ratio_aug']=(R_.sep+0.5)/(R_.aug*N['sep']/N['aug']+0.5)
exp_=R_.sep*N['oct']/N['sep']; R_['z']=(R_.oct-exp_)/np.sqrt(exp_+R_.sep*(N['oct']/N['sep'])**2+1)
R_['k']=R_.fields.str.count('&')+1
R_.to_parquet('/tmp/hunt2/cells.parquet'); print(len(R_), N)
