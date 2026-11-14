from base import *
import sys
# rough expected-macro-F1 calculator: per class T (true test count, est. from marker scaling), base precision p0, flipped-row precision pf
T={'mule_account':2000,'velocity_bot':1000,'layering_chain':861,'sim_swap':492,'account_takeover':785,'phishing_collect':868,'qr_swap':1100,
   'refund_abuse':431,'fake_cashback':807,'loan_app_extortion':572,'merchant_collusion':468}
def macro_delta(base, cand, pf, p0=0.8, Tm=T):
    b=base.label.values; c=cand.label.values; ch=b!=c; tot=0; rows=[]
    for k in Tm:
        n0=(b==k).sum(); tp0=p0*n0
        add=((c==k)&ch).sum(); rem=((b==k)&ch).sum()
        n1=n0+add-rem; tp1=tp0+pf*add-p0*rem
        f0=2*tp0/(n0+Tm[k]); f1=2*tp1/(n1+Tm[k]); tot+=f1-f0; rows.append((k,add,rem,round(f1-f0,4)))
    return tot/12, rows
if __name__=='__main__':
    b=pd.read_csv(sys.argv[1]); c=pd.read_csv(sys.argv[2])
    for pf in [float(x) for x in sys.argv[3].split(',')]:
        d,r=macro_delta(b,c,pf); print('pf',pf,'macro delta',round(d,4)); 
    print(r)
