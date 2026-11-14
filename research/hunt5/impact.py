import pandas as pd, numpy as np
R='/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/'
T={'account_takeover':778,'fake_cashback':597,'layering_chain':1186,'loan_app_extortion':380,'merchant_collusion':499,'mule_account':2091,'phishing_collect':1195,'qr_swap':1328,'refund_abuse':586,'sim_swap':755,'velocity_bot':1236}  # ceil2 T2 (whole test)
b=pd.read_csv(R+'submissions/v10_v6_octrules.csv').set_index('id').label
def score(sub, chg_ok):
    """base rows precision p0; changed rows correct with prob chg_ok; old label of changed rows assumed wrong w.p. chg_ok"""
    c=sub!=b; f=0
    for k,t in T.items():
        p0=0.85
        M=(sub==k).sum(); keep=((sub==k)&~c).sum(); new=((sub==k)&c).sum()
        tp=p0*keep+chg_ok*new
        f+=2*tp/(M+t)
    return f
for name in ['v11_blockseg']:
    s=pd.read_csv(R+f'submissions/{name}.csv').set_index('id').label.reindex(b.index)
    for ok in [0.7,0.8,0.9]:
        # base: rows that change were correct with prob (1-ok) in base
        base=0; c=s!=b
        for k,t in T.items():
            M=(b==k).sum(); keep=((b==k)&~c).sum(); old=((b==k)&c).sum()
            base+=2*(0.85*keep+(1-ok)*old)/(M+t)
        print(name,'flip-precision',ok,'macro gain %.4f'%((score(s,ok)-base)/12))
