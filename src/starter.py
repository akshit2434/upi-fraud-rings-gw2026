"""Organizers' starter notebook, verbatim logic. Val Jul-Aug -> Sep, then full fit -> submissions/starter.csv"""
import numpy as np, pandas as pd, sys
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, classification_report
DATA = 'data/upi-fraud-rings'
train = pd.read_csv(f'{DATA}/train.csv'); test = pd.read_csv(f'{DATA}/test.csv'); accounts = pd.read_csv(f'{DATA}/accounts.csv')
home = accounts.set_index('account_id').home_city
CATS = ['receiver_type', 'txn_type', 'merchant_category', 'app']
def row_features(d):
    X = pd.DataFrame(index=d.index); t = pd.to_datetime(d.timestamp)
    X['hour'] = t.dt.hour; X['weekday'] = t.dt.weekday
    for c in ['amount', 'sim_age_hours', 'device_age_days', 'sender_account_age_days', 'new_payee', 'pin_retries', 'login_to_txn_sec', 'balance_before', 'qr_vpa_mismatch']:
        X[c] = d[c]
    X['amount_share_of_balance'] = d.amount / d.balance_before
    X['away_from_home'] = (d.device_city != d.sender_id.map(home)).astype(int)
    for c in CATS:
        X[c] = pd.Categorical(d[c].fillna('none'), categories=sorted(set(train[c].fillna('none')) | set(test[c].fillna('none')))).codes
    return X
X, Xt = row_features(train), row_features(test)
is_sep = train.timestamp.str[:7] == '2026-09'
model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
model.fit(X[~is_sep], train.label[~is_sep]); pred = model.predict(X[is_sep])
print('starter val macro-F1:', round(f1_score(train.label[is_sep], pred, average='macro'), 4))
print(classification_report(train.label[is_sep], pred, digits=3, zero_division=0))
model.fit(X, train.label)
sub = pd.DataFrame({'id': test.txn_id, 'label': model.predict(Xt)}); sub.to_csv('submissions/starter.csv', index=False)
print(sub.label.value_counts().to_dict())
