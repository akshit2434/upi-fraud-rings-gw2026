from load import *
pd.set_option('display.width',250)
tr=pd.read_csv(R+'data/upi-fraud-rings/train.csv'); te=pd.read_csv(R+'data/upi-fraud-rings/test.csv')
tr=tr[tr.timestamp>='2026-08-01']
cols=[c for c in te.columns if c not in ('txn_id','timestamp')]
for nm,x in [('train AugSep',tr),('test',te)]:
    print(nm,'nulls', x[cols].isna().mean().round(4)[lambda s:s>0].to_dict())
# null patterns per txn_type x receiver_type
def pat(x):
    k=x.txn_type+'|'+x.receiver_type+'|cat:'+x.merchant_category.isna().map({True:'na',False:'y'})+'|mis:'+x.qr_vpa_mismatch.fillna(-1).astype(int).astype(str)
    return k.value_counts()
a=pat(tr)/len(tr)*1e4; b=pat(te)/len(te)*1e4
t=pd.concat([a.rename('trn10k'),b.rename('test10k')],axis=1).fillna(0); t['ratio']=(t.test10k+.01)/(t.trn10k+.01)
print(t.round(2).sort_values('ratio').to_string())
# fraud share in train per pattern
k=tr.txn_type+'|'+tr.receiver_type+'|cat:'+tr.merchant_category.isna().map({True:'na',False:'y'})+'|mis:'+tr.qr_vpa_mismatch.fillna(-1).astype(int).astype(str)
print((tr.label!='legit').groupby(k).mean().round(3).to_dict())
