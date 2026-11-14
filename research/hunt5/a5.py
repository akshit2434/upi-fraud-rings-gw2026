import pandas as pd, numpy as np
df=pd.read_parquet('/tmp/hunt5/rawj.parquet')
M=df[df.receiver_type=='MERCHANT'].copy()
tr=M[M.is_test==0].copy(); tr['f']=tr.label!='legit'; tr['I']=tr.f&~tr.conn
tr['b']=pd.qcut(np.log(tr.rec_cnt),20,duplicates='drop')
print(tr.groupby('b').agg(n=('f','size'),fr=('f','mean'),iso=('I','mean'),nm=('receiver_id','nunique')).round(4))
# merchant-level: distribution of merchant counts; 
mc=M.groupby('receiver_id').size()
print(mc.describe(percentiles=[.1,.25,.5,.75,.9,.99]))
print('n merchants used', len(mc), 'of 30000')
# onboarded
tr['onb']=pd.to_datetime(tr.m_onboarded_on); tr['ob']=pd.qcut(tr.onb.astype('int64'),10,duplicates='drop')
print(tr.groupby('ob').agg(fr=('f','mean'),iso=('I','mean')).round(4))
print(tr.groupby('m_category').agg(n=('f','size'),fr=('f','mean'),iso=('I','mean'),cnt=('rec_cnt','median')).round(4))
