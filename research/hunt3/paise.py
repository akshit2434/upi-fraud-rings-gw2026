from base import *
d=load(); d=d[d.grp!='aug'].copy(); d['lab']=np.where(d.grp=='oct',d.v9,d.label)
pd.set_option('display.width',250,'display.max_columns',30,'display.max_rows',300)
d['int']=(d.amount.round(2)==d.amount.round(0))
d['r100']=d.amount%100==0
d['ib']=np.where(d.r100,'r100',np.where(d.int,'int','paise'))
d['seg']=d.txn_type+'>'+d.receiver_type
t=d.groupby(['seg','ib','grp']).agg(n=('lab','size'),fr=('lab',lambda s:(s!='legit').mean())).unstack(2)
t['ratio']=t[('n','oct')]/t[('n','sep')]
print(t.round(3))
# train: legit paise rate by seg
tr=d[d.grp=='sep']; print(tr.groupby(['seg',tr.label!='legit']).int.mean().unstack().round(3))
