from base import *
d=load(); s=pd.read_parquet('sepp.parquet')[['txn_id','sp','anom','mk']]; d=d.merge(s,on='txn_id')
d['pred']=np.where(d.grp=='oct',d.v9,d.sp)
d['night']=d.hr.isin([23,0,1,2,3,4,5]); d['eve']=d.hr.isin([19,20,21,22])
d['simfresh']=d.sim_age_hours<72; d['devnew']=d.device_age_days<2; d['pin']=d.pin_retries>0
d['P2P']=d.receiver_type=='P2P'; d['login_fast']=d.login_to_txn_sec<15
d['has_rev']=d.rev_dt.notna()&(d.rev_dt.abs()<86400*7) if d.rev_dt.dtype!=object else False
d['tiny_in']=d.s_tiny_in_dt.notna()
d['onward']=d.r_out_next1h>0
FE=['P2P','night','eve','simfresh','devnew','device_is_own','away_from_home','new_payee','pin','login_fast','amt_round100','amt_round1000','amt_just_under','amt_le10','onward','tiny_in']
NUM=['amount','amt_over_bal','login_to_txn_sec','s_n_1h','s_n_1d','d_nacc_1h','r_nsend_1h','r_nsend_1d','s2_7d_ref','s2_7d_lend','s2_7d_late','p2_7d_n','r2_7d_round','r_nround_senders_7d']
pd.set_option('display.width',250,'display.max_columns',40,'display.max_rows',500)
out=[]
for c in LABELS[1:]:
    for g,name in [('sep','S_true'),('oct','O_pred')]:
        x=d[(d.grp==g)&d.mk&((d.label==c) if g=='sep' else (d.v9==c))]
        r={'cls':c,'set':name,'n':len(x)}
        for f in FE: r[f]=x[f].astype(float).mean()
        for f in NUM: r[f]=x[f].median()
        r['type']=x.txn_type.value_counts(normalize=True).round(2).head(3).to_dict()
        r['catg']=x.cat.value_counts(normalize=True).round(2).head(3).to_dict()
        out.append(r)
    # also Oct unmarked predicted
t=pd.DataFrame(out).set_index(['cls','set'])
print(t[['n']+FE].round(2)); print(t[NUM].round(2)); 
for i,r in t.iterrows(): print(i, r['type'], r['catg'])
