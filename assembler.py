#


#
import random
import numpy
import joblib
import pandas
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef

import torch

# from statsmodels.tsa.stattools import adfuller

from xgboost import XGBClassifier

from sklearn.feature_selection import mutual_info_classif, RFECV
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, kendalltau
#


#
data = pandas.read_excel('./data/dataset_refined.xlsx')
data['datetime'] = pandas.to_datetime(data['datetime'], infer_datetime_format=True)
data = data.set_index('datetime')


def lag(data, n_lags):
    data_lagged = data.copy()
    cols = data.columns.values
    for j in range(n_lags):
        data_lagged[[x + '_LAG{0}'.format(j + 1) for x in data.columns.values]] = data[cols].shift(periods=(j + 1))
    return data_lagged


# data = lag(data=data, n_lags=40)
data = data.dropna()

removables = []
# removables = ['FLAG']

targets = ['_TARGET', 'TARGET_1M', 'TARGET_3M', 'TARGET_6M', '_TARGET_ewm21', '_TARGET_ewm55']
x_factors = [x for x in data.columns if not any([y in x for y in targets + removables])]

X = data[x_factors].values.astype(dtype=float)
Y = (data[targets[1]] > data[targets[0]]).values.astype(dtype=int)
Z = data[['_TARGET', '_TARGET_ewm21', '_TARGET_ewm55']].values.astype(dtype=float)

ix_start = pandas.to_datetime('2006-12-27')
ix_thresh = pandas.to_datetime('2014-07-25')
ix_end = pandas.to_datetime('2021-06-27')

data = data[(data.index >= ix_start) * (data.index <= ix_end)]
thresh_val = numpy.where(data.index.values == ix_thresh)[0][0]

X_train, X_test = X[:thresh_val], X[thresh_val:]
Y_train, Y_test = Y[:thresh_val], Y[thresh_val:]
Z_train, Z_test = Z[:thresh_val], Z[thresh_val:]


# ex12
"""
alpha = 0.05
values = numpy.array([kendalltau(x=X_train[:, j], y=Y_train)[1] for j in range(X_train.shape[1])])
fs_mask = values <= alpha

X_train = X_train[:, fs_mask]
X_test = X_test[:, fs_mask]
"""
# """
# """
fs_mask_name = './fsmask_GRU100.pkl'
fs_mask = joblib.load(fs_mask_name)

X_train = X_train[:, fs_mask]
X_test = X_test[:, fs_mask]

scaler_name = './scaler_GRU100.pkl'
scaler = joblib.load(scaler_name)

X_train_ = scaler.transform(X_train)
X_test_ = scaler.transform(X_test)
# '''
projector_name = './projector_GRU100.pkl'
projector = joblib.load(projector_name)

X_train_ = projector.transform(X_train_)
X_test_ = projector.transform(X_test_)
# '''
# '''
Y_train_ = numpy.concatenate(((Y_train.reshape(-1, 1) == 0).astype(dtype=int),
                              (Y_train.reshape(-1, 1) == 1).astype(dtype=int)),
                             axis=1)
Y_test_ = numpy.concatenate(((Y_test.reshape(-1, 1) == 0).astype(dtype=int),
                             (Y_test.reshape(-1, 1) == 1).astype(dtype=int)),
                            axis=1)

window = 100

xx_train = []
xx_val = []
yy_train = []
yy_val = []
for j in range(X_train_.shape[0] - window + 1):
    xx_train.append(X_train_[j:j + window, :].reshape(1, window, X_train_.shape[1]))
    # yy_train.append(Y_train_[j:j + window].reshape(1, window, 2))
for j in range(X_test_.shape[0] - window + 1):
    xx_val.append(X_test_[j:j + window, :].reshape(1, window, X_test_.shape[1]))
    # yy_val.append(Y_test_[j:j + window].reshape(1, window, 2))
xx_train_ = numpy.concatenate(xx_train, axis=0)
xx_val_ = numpy.concatenate(xx_val, axis=0)
# yy_train_ = numpy.concatenate(yy_train, axis=0)
# yy_val_ = numpy.concatenate(yy_val, axis=0)
yy_train_ = Y_train_[(window - 1):, :]
yy_val_ = Y_test_[(window - 1):, :]

X_train_ = torch.tensor(xx_train_, dtype=torch.float)
Y_train_ = torch.tensor(yy_train_, dtype=torch.float)
X_test_ = torch.tensor(xx_val_, dtype=torch.float)
Y_test_ = torch.tensor(yy_val_, dtype=torch.float)
# '''

model_name = './model_GRU100.pkl'
model = joblib.load(model_name)
y_hat_train = model.predict(X=X_train_)
y_hat_test = model.predict(X=X_test_)
# '''
y_hat_train = y_hat_train[:, 0] < y_hat_train[:, 1]
y_hat_test = y_hat_test[:, 0] < y_hat_test[:, 1]


cm_train = confusion_matrix(y_true=Y_train[(window - 1):], y_pred=y_hat_train)
cm_test = confusion_matrix(y_true=Y_test[(window - 1):], y_pred=y_hat_test)

ac_train = accuracy_score(y_true=Y_train[(window - 1):], y_pred=y_hat_train)
ac_test = accuracy_score(y_true=Y_test[(window - 1):], y_pred=y_hat_test)

matt_train = matthews_corrcoef(y_true=Y_train[(window - 1):], y_pred=y_hat_train)
matt_test = matthews_corrcoef(y_true=Y_test[(window - 1):], y_pred=y_hat_test)
# '''
'''
cm_train = confusion_matrix(y_true=Y_train, y_pred=y_hat_train)
cm_test = confusion_matrix(y_true=Y_test, y_pred=y_hat_test)

ac_train = accuracy_score(y_true=Y_train, y_pred=y_hat_train)
ac_test = accuracy_score(y_true=Y_test, y_pred=y_hat_test)

matt_train = matthews_corrcoef(y_true=Y_train, y_pred=y_hat_train)
matt_test = matthews_corrcoef(y_true=Y_test, y_pred=y_hat_test)
'''
print(cm_train, '\n', cm_test, '\n',
      '{0:.4f}'.format(ac_train), '\n', '{0:.4f}'.format(matt_train), '\n',
      '{0:.4f}'.format(ac_test), '\n', '{0:.4f}'.format(matt_test), '\n')


def cc(mx):
    return (mx[1, 1] + mx[2, 2]) / (mx[1, 1] + mx[1, 2] + mx[2, 1] + mx[2, 2])


Z_train = Z_train[(window - 1):]
Z_test = Z_test[(window - 1):]

# """

# y_hat_train = Y_train
# y_hat_test = Y_test
# y_hat_train = numpy.ones(shape=(Y_train.shape[0],))
# y_hat_test = numpy.ones(shape=(Y_test.shape[0],))


# signal_train = (Z_train[:, 1] > Z_train[:, 2]).astype(dtype=int)
# signal_test = (Z_test[:, 1] > Z_test[:, 2]).astype(dtype=int)
signal_train = numpy.ones(shape=(Z_train.shape[0]),)
signal_test = numpy.ones(shape=(Z_test.shape[0]),)


def find_touch(a, b):
    a_shift = pandas.Series(a).shift(1).values
    b_shift = pandas.Series(b).shift(1).values
    compare = (a > b).astype(dtype=int)
    shift_compare = (a_shift > b_shift).astype(dtype=int)
    result = (compare != shift_compare).astype(dtype=int)
    result[0] = 0
    result[-1] = 0
    return result

"""
p = 0.005
touch_train = (((Z_train[:, 0] / Z_train[:, 1]) <= (1 + p)) * ((Z_train[:, 0] / Z_train[:, 1]) >= (1 - p))).astype(dtype=int)
touch_test = (((Z_test[:, 0] / Z_test[:, 1]) <= (1 + p)) * ((Z_test[:, 0] / Z_test[:, 1]) >= (1 - p))).astype(dtype=int)
"""
# """
# touch_train = find_touch(Z_train[:, 0], Z_train[:, 1])
# touch_test = find_touch(Z_test[:, 0], Z_test[:, 1])
touch_train = numpy.ones(shape=(Z_train.shape[0]),)
touch_test = numpy.ones(shape=(Z_test.shape[0]),)
# """

# THE ALGO

def calcatt(y_hat, z, signal, touch):
    status = 0
    yield_cum = 1
    stop_loss, take_profit = numpy.nan, numpy.nan
    
    deals = []
    statuses = []
    yield_cums = []
    for j in range(y_hat.shape[0]):
        if status == 1:
            if (z[j, 0] >= take_profit) or (z[j, 0] <= stop_loss):
                status = 0
                deal[1] = z[j, 0]
                deal[2] = deal[1] / deal[0]
                if z[j, 0] >= take_profit:
                    deal[3] = 'TP'
                else:
                    deal[3] = 'SL'
                yield_cum = yield_cum * deal[2]
                deals.append(deal)
        else:
            if (y_hat[j] == 1) and (signal[j] == 1) and (touch[j] == 1):
                status = 1
                deal = [numpy.nan, numpy.nan, numpy.nan, 'NA']
                deal[0] = z[j, 0]
                stop_loss = z[j, 0] * 0.9975
                take_profit = z[j, 0] * 1.0050
        statuses.append(status)
        yield_cums.append(yield_cum)
        
    return deals, statuses, yield_cums


# go

deals_train, statuses_train, yield_cums_train = calcatt(y_hat_train, Z_train, signal_train, touch_train)
deals_test, statuses_test, yield_cums_test = calcatt(y_hat_test, Z_test, signal_test, touch_test)

dyno_train = [[y_hat_train[j]] + [signal_train[j]] + [touch_train[j]] + [statuses_train[j]] + [yield_cums_train[j]] for j in range(y_hat_train.shape[0])]
dyno_test = [[y_hat_test[j]] + [signal_test[j]] + [touch_test[j]] + [statuses_test[j]] + [yield_cums_test[j]] for j in range(y_hat_test.shape[0])]

dyno_train = pandas.DataFrame(data=dyno_train, columns=['MACRO', 'SIGNAL', 'TOUCH', 'STATUS', 'CUM'])
dyno_test = pandas.DataFrame(data=dyno_test, columns=['MACRO', 'SIGNAL', 'TOUCH', 'STATUS', 'CUM'])

deals_train = pandas.DataFrame(data=deals_train, columns=['BUY', 'SELL', 'YIELD', 'ACTION'])
deals_test = pandas.DataFrame(data=deals_test, columns=['BUY', 'SELL', 'YIELD', 'ACTION'])


hag_train = pandas.concat((dyno_train, deals_train), axis=1)
hag_test = pandas.concat((dyno_test, deals_test), axis=1)


view_train = {'signal': (signal_train == 1).astype(dtype=int),
              'touch': (touch_train == 1).astype(dtype=int),
              'ewm21': data['_TARGET_ewm21'].values[:thresh_val][(window - 1):],
              'ewm55': data['_TARGET_ewm55'].values[:thresh_val][(window - 1):],
              'quote_curr': data[targets[0]].values[:thresh_val][(window - 1):],
              'quote_1mfw': data[targets[1]].values[:thresh_val][(window - 1):]}
view_test = {'signal': (signal_test == 1).astype(dtype=int),
              'touch': (touch_test == 1).astype(dtype=int),
             'ewm21': data['_TARGET_ewm21'].values[thresh_val:][(window - 1):],
             'ewm55': data['_TARGET_ewm55'].values[thresh_val:][(window - 1):],
              'quote_curr': data[targets[0]].values[thresh_val:][(window - 1):],
              'quote_1mfw': data[targets[1]].values[thresh_val:][(window - 1):]}


view_train = pandas.DataFrame(view_train)
view_test = pandas.DataFrame(view_test)


deals_train.to_excel('./deals_train.xlsx')
deals_test.to_excel('./deals_test.xlsx')
dyno_train.to_excel('./dyno_train.xlsx')
dyno_test.to_excel('./dyno_test.xlsx')

fig, ax = pyplot.subplots()
dyno_train['CUM'].plot(kind='line', ax=ax)
axx = ax.twinx()
dyno_train['STATUS'].plot(kind='bar', ax=axx)
ax.set_title("Train")
fig.savefig("cum_yield_train.png")
pyplot.close(fig)

fig, ax = pyplot.subplots()
dyno_test['CUM'].plot(kind='line', ax=ax)
axx = ax.twinx()
dyno_test['STATUS'].plot(kind='bar', ax=axx)
fig.savefig("cum_yield_test.png")
pyplot.close(fig)


def sharpe_ratio(ra, rf):
    return (ra - rf).mean() / (ra - rf).std()


ra_train = dyno_train['CUM'].pct_change().dropna()
ra_test = dyno_test['CUM'].pct_change().dropna()
sr_train = sharpe_ratio(ra_train, numpy.zeros(shape=ra_train.shape))
sr_test = sharpe_ratio(ra_test, numpy.zeros(shape=ra_test.shape))

"""
from matplotlib import pyplot
da = view_train.iloc[30:40, :]
fig, ax = pyplot.subplots(3, 1)
da[['quote_curr', 'quote_1mfw', 'ewm21', 'ewm55']].plot(ax=ax[0])
da[['signal']].plot(kind='bar', ax=ax[1])
da[['touch']].plot(kind='bar', ax=ax[2])
fig.savefig('C:/Users/azizove/Desktop/train.png')
"""
#
"""
hah = pandas.concat((pandas.DataFrame(data=Z_train, columns=['_TARGET', '_TARGET_ewm21', '_TARGET_ewm55']),
                     pandas.Series(touch_train, name='touch')), axis=1)
HAHA = hah.iloc[30:40, :]
"""