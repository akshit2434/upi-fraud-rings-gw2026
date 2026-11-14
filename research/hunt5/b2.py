import pandas as pd, numpy as np
R='/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/'
T0={'account_takeover':778,'fake_cashback':597,'layering_chain':1186,'loan_app_extortion':380,'merchant_collusion':499,'mule_account':2091,'phishing_collect':1195,'qr_swap':1328,'refund_abuse':586,'sim_swap':755,'velocity_bot':1236}
rd=lambda f: pd.read_csv(R+'submissions/'+f).set_index('id').label
v6=pd.Series(np.load(R+'preds/v6_R1_bag_test_P.npy').argmax(1)) # placeholder not used
subs={'v9':rd('v9_v6_qrrule.csv'),'v10':rd('v10_v6_octrules.csv'),'v11':rd('v11_blockseg.csv')}
idx=subs['v10'].index; subs={k:v.reindex(idx) for k,v in subs.items()}
obs={('v9','v10'):0.0111,('v10','v11'):0.0150}
def gain(a,b,ok,p0,sc):
    c=a!=b; g=0
    for k,t in T0.items():
        t*=sc
        Ma=(a==k).sum(); Mb=(b==k).sum()
        tpa=p0*((a==k)&~c).sum()+(1-ok)*((a==k)&c).sum()
        tpb=p0*((b==k)&~c).sum()+ok*((b==k)&c).sum()
        g+=2*tpb/(Mb+t)-2*tpa/(Ma+t)
    return g/12
print('sc ok p0 | v9->v10  v10->v11')
for sc in [0.5,0.6,0.7,0.8,1.0]:
    for ok in [0.85,0.95]:
        print(sc,ok,0.85,'| %.4f %.4f'%(gain(subs['v9'],subs['v10'],ok,0.85,sc),gain(subs['v10'],subs['v11'],ok,0.85,sc)))
print('observed 0.0111 0.0150')
