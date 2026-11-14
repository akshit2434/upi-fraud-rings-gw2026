from base import *
import itertools
d=load(); d=d[d.grp!='aug'].copy()
d['mk']=(d.txn_type!='QR')&d.qr_vpa_mismatch.notna()
d['nonqr']=d.txn_type!='QR'
d['catg']=d.cat
d['hb']=pd.cut(d.hr,[-1,4,8,12,16,19,23],labels=['00-04','05-08','09-12','13-16','17-19','20-23']).astype(str)
d['ab']=pd.cut(d.amount,[0,10,100,500,2000,5000,20000,1e9],labels=['<=10','<=100','<=500','<=2k','<=5k','<=20k','>20k']).astype(str)
d['ob']=pd.cut(d.amt_over_bal.fillna(0),[-1,0.3,0.7,0.95,1e9],labels=['<.3','<.7','<.95','>=.95']).astype(str)
d['sim']=np.where(d.sim_age_hours<72,'simfresh','simold'); d['dev']=np.where(d.device_age_days<2,'devnew','devold')
d['own']=np.where(d.device_is_own==1,'own','other'); d['away']=np.where(d.away_from_home==1,'away','home')
d['np']=np.where(d.new_payee==1,'newp','oldp'); d['pin']=np.where(d.pin_retries>0,'pin','nopin')
d['lg']=pd.cut(d.login_to_txn_sec,[-1,10,30,1e9],labels=['lg<=10','lg<=30','lg>30']).astype(str)
d['onw']=np.where(d.r_out_next1h>0,'onward','noonw'); d['fan']=np.where(d.r_nsend_1h>=2,'fanin','nofan')
d['rnd']=np.where(d.amt_round1000==1,'r1000',np.where(d.amt_round100==1,'r100',np.where(d.amt_just_under==1,'jund','plain')))
F=['txn_type','receiver_type','catg','hb','ab','ob','sim','dev','own','away','np','pin','lg','onw','fan','rnd']
d=d[d.nonqr]
d['v9f']=(d.v9.fillna('legit')!='legit'); d['tf']=d.label.notna()&(d.label!='legit')
S=d[d.grp=='sep']; O=d[d.grp=='oct']
res=[]
for k in [1,2]:
  for combo in itertools.combinations(F,k):
    a=S.groupby(list(combo)).agg(Sn=('mk','size'),Smk=('mk','sum'),Stf=('tf','sum'))
    b=O.groupby(list(combo)).agg(On=('mk','size'),Omk=('mk','sum'),Ov9=('v9f','sum'),Ov9u=('v9f',lambda s: 0))
    # v9 fraud among unmarked
    b['Ov9_unm']=O[~O.mk].groupby(list(combo)).v9f.sum()
    t=a.join(b,how='outer').fillna(0); t['combo']='|'.join(combo); t.index=[ '|'.join(map(str,i)) if isinstance(i,tuple) else str(i) for i in t.index]
    res.append(t)
t=pd.concat(res)
tot_s=S.mk.sum(); tot_o=O.mk.sum(); print('marked totals sep51',tot_s,'oct',tot_o)
t['exp_Omk']=t.Smk*tot_o/tot_s
t['z']=(t.Omk-t.exp_Omk)/np.sqrt(t.exp_Omk+1)
# expected true fraud in Oct cell (unmarked) from Sep ratio unmarked fraud / marked
t['Stf_unm']=t.Stf-t.Smk  # approx (marked legit tiny)
t['est_O_unm_fraud']=t.Omk*(t.Stf_unm/t.Smk.replace(0,np.nan))
t['gap']=t.est_O_unm_fraud-t.Ov9_unm
pd.set_option('display.width',250,'display.max_rows',100)
cols=['Sn','Smk','Stf','On','Omk','exp_Omk','z','Ov9_unm','est_O_unm_fraud','gap']
print('--- marked excess in Oct'); print(t[t.Omk>=15].sort_values('z',ascending=False)[cols].head(40).round(1))
print('--- marked deficit'); print(t[t.exp_Omk>=15].sort_values('z')[cols].head(25).round(1))
t.to_pickle('mcells.pkl')
