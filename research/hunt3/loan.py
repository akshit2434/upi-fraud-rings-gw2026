from base import *
d=load()
pd.set_option('display.width',250,'display.max_columns',30,'display.max_rows',200)
d['eve']=d.hr.isin([20,21,22,23])
for g in ['sep','oct']:
    x=d[(d.grp==g)]
    L=x[x.cat=='lending']
    L=L.assign(e=L.eve)
    sc=L.groupby('sender_id').agg(n=('e','size'),ne=('e','sum'))
    x=x.merge(sc,left_on='sender_id',right_index=True,how='left')
    L=x[x.cat=='lending']
    lab='label' if g=='sep' else 'v9'
    print(g, 'lending rows by (sender lending count, evening)')
    t=L.groupby([L.n.clip(upper=5),L.eve])[lab].agg(lambda s: (s=='loan_app_extortion').mean()).unstack()
    c=L.groupby([L.n.clip(upper=5),L.eve]).size().unstack()
    print(pd.concat([c,t.round(2)],axis=1,keys=['n','loanrate']))
    # P2P rows of senders with >=2 evening lending payments
    hv=x[(x["ne"]>=2)]
    print(' rows of senders with >=2 eve lending', len(hv), hv[lab].value_counts().head(6).to_dict())
    print(pd.crosstab([hv.cat=='lending',hv.eve],hv[lab]))
