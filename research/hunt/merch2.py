from load import *
df=load()
tr=df[df.is_test==0].copy()
M=tr[tr.receiver_type=='MERCHANT'].copy()
M['pop']=M.groupby('receiver_id').txn_id.transform('size')
M['isostruct']=(M.grp=='iso')&M.label.isin(STRUCT)
M['iso']=(M.grp=='iso'); M['fraud']=M.label!='legit'
M['bin']=pd.qcut(M['pop'],20,duplicates='drop')
print(M.groupby('bin').agg(n=('txn_id','size'),iso_struct=('isostruct','mean'),iso=('iso','mean'),fraud=('fraud','mean')).round(4).to_string())
mp=M.groupby('receiver_id').agg(pop=('txn_id','size'),fr=('fraud','mean'))
print('merchants w/ any txn', len(mp), 'of 30000'); print(mp['pop'].describe(percentiles=[.01,.05,.1,.25,.5,.75,.9,.99]))
