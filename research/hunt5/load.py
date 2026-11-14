import pandas as pd, numpy as np
D='/Users/akshit2434/.t3/scratch/2026-11-13-https-hack4hack-com-hackathons-ml-a175759f/data/'
def load():
    tr=pd.read_csv(D+'upi-fraud-rings/train.csv'); te=pd.read_csv(D+'upi-fraud-rings/test.csv')
    conn=pd.read_parquet(D+'conn.parquet')
    tr=tr.merge(conn,on='txn_id',how='left'); tr['conn']=tr.conn.fillna(False)
    te['label']=None; te['conn']=False
    tr['is_test']=0; te['is_test']=1
    df=pd.concat([tr,te],ignore_index=True)
    df['ts']=pd.to_datetime(df.timestamp); df['t']=(df.ts-pd.Timestamp('2026-07-01')).dt.total_seconds()
    return df
