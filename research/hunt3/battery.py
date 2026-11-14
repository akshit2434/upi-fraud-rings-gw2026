from base import *
d=load(); d=d[d.grp!='aug'].copy(); d['lab']=np.where(d.grp=='oct',d.v9,d.label)
pd.set_option('display.width',250,'display.max_columns',30,'display.max_rows',300)
a=d.amount; P2P=d.receiver_type=='P2P'; M=~P2P; night=d.hr.isin([0,1,2,3,4,5]); eve=d.hr.isin([19,20,21,22,23])
ju=(a%1000==999)|(a%1000==990)|(a%500==499)
mk={
 'mule: P2P fan-in>=3 senders/1d': P2P&(d.r_nsend_1d>=3),
 'mule: P2P fan-in>=2/1h': P2P&(d.r_nsend_1h>=2),
 'mule/layer: recv onward<=1h': P2P&(d.r_out_next1h>0),
 'layer: amt>20k P2P onward<=1h': P2P&(a>20000)&(d.r_out_next1h>0),
 'layer: incoming<=1h before (s_in_prev1h)': (d.s_in_prev1h>0)&P2P,
 'velocity: device>=3 accts 1h': d.d_nacc_1h>=3,
 'velocity: device>=5 rows 1h': d.d_n_1h>=5,
 'velocity: ecom/gaming non-own login<=10s': d.cat.isin(['ecommerce','gaming'])&(d.device_is_own==0)&(d.login_to_txn_sec<=10),
 'sim: sim<24h': d.sim_age_hours<24,
 'sim: sim<72h & amt/bal>.7': (d.sim_age_hours<72)&(d.amt_over_bal>0.7),
 'sim: night P2P amt/bal>.7': night&P2P&(d.amt_over_bal>0.7),
 'ato: new device<1d & away': (d.device_age_days<1)&(d.away_from_home==1),
 'ato: pin>=1 & non-own device': (d.pin_retries>=1)&(d.device_is_own==0),
 'ato: non-own device P2P amt/bal>.7': (d.device_is_own==0)&P2P&(d.amt_over_bal>0.7),
 'phish: COLLECT just-under': (d.txn_type=='COLLECT')&ju,
 'phish: COLLECT new payee >=1000': (d.txn_type=='COLLECT')&(d.new_payee==1)&(a>=1000),
 'qr: QR->P2P (any)': (d.txn_type=='QR')&P2P,
 'qr: QR mismatch=1': (d.txn_type=='QR')&(d.qr_vpa_mismatch==1),
 'refund: REFUND': d.txn_type=='REFUND',
 'refund: sender >=2 refunds 7d': (d.txn_type=='REFUND')&(d.s2_7d_ref>=2),
 'cashback: tiny<=10 P2P PAY': (a<=10)&P2P&(d.txn_type=='PAY'),
 'cashback: tiny-in before (s_tiny_in_dt)': d.s_tiny_in_dt.notna(),
 'loan: lending': d.cat=='lending',
 'loan: lending evening': (d.cat=='lending')&eve,
 'loan: sender >=2 lending 7d': (d.s2_7d_lend>=2),
 'collusion: merchant round1000': M&(d.amt_round1000==1),
 'collusion: recv >=3 round senders 7d': M&(d.r_nround_senders_7d>=3),
 'any: nonQR mismatch set': (d.txn_type!='QR')&d.qr_vpa_mismatch.notna(),
 'any: reverse pair payment (rev_dt)': d.rev_dt.notna(),
}
rows=[]
for k,m in mk.items():
    s=d[m&(d.grp=='sep')]; o=d[m&(d.grp=='oct')]
    rows.append(dict(marker=k,sep51=len(s),oct=len(o),ratio=len(o)/max(len(s),1),sep_true_fr=(s.lab!='legit').mean(),oct_v9_fr=(o.lab!='legit').mean(),
       sep_top=s.lab[s.lab!='legit'].value_counts().head(2).to_dict(),oct_top=o.lab[o.lab!='legit'].value_counts().head(2).to_dict()))
t=pd.DataFrame(rows).set_index('marker'); print(t.round(2).to_string()); t.to_csv('battery.csv')
