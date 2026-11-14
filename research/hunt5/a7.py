import pandas as pd, numpy as np
df=pd.read_parquet('/tmp/hunt5/raw.parquet')
tr=df[df.is_test==0].copy()
tr['f']=tr.label!='legit'
# account as receiver: counts of incoming rows, incoming fraud rows (by label)
inc=tr.groupby('receiver_id').agg(in_n=('f','size'),in_f=('f','sum'))
inc_lab=tr[tr.f].groupby(['receiver_id','label']).size().unstack(fill_value=0)
out=tr.groupby('sender_id').agg(out_n=('f','size'),out_f=('f','sum'))
tr=tr.join(inc,on='sender_id').join(out.rename(columns={'out_n':'r_out_n','out_f':'r_out_f'}),on='receiver_id')
for c in ['in_n','in_f','r_out_n','r_out_f']: tr[c]=tr[c].fillna(0)
tr['grp']=np.where(tr.label=='legit','legit',np.where(tr.conn,'C','I'))
print(tr.groupby(['grp']).apply(lambda x:pd.Series({'sender_recv_any':(x.in_n>0).mean(),'sender_recv_fraud':(x.in_f>0).mean(),'recv_sends_any':(x.r_out_n>0).mean(),'recv_sends_fraud':(x.r_out_f>0).mean()})).round(4))
X=tr[tr.grp!='C']
print(X.groupby('label').apply(lambda x:pd.Series({'s_recv_any':(x.in_n>0).mean(),'s_recv_fraud':(x.in_f>0).mean(),'r_sends_fraud':(x.r_out_f>0).mean(),'n':len(x)})).round(4))
