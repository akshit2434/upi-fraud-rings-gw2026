from load import *
df=load(); tr=df[df.is_test==0]
tr=tr.assign(cat=tr.merchant_category.fillna('P2P')+'|'+tr.txn_type)
leg=tr[tr.grp=='legit'].cat.value_counts(normalize=True)
for cls in STRUCT+['fake_cashback']:
    for g in ['iso','conn']:
        s=tr[(tr.label==cls)&(tr.grp==g)].cat.value_counts(normalize=True)
        d=(s.reindex(leg.index.union(s.index),fill_value=0)-leg.reindex(leg.index.union(s.index),fill_value=0)).sort_values()
        print(cls,g,' +:',', '.join(f'{k} {v:+.3f}' for k,v in d.tail(5)[::-1].items()),' -:',', '.join(f'{k} {v:+.3f}' for k,v in d.head(3).items()))
