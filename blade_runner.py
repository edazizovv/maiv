#


#
import random
import numpy
import pandas
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef

import torch
from torch import nn

# from merrill_model.neuro.neura import WrappedNN
from neura import WrappedNN
from mayer.the_skeleton.func import make_params

# from statsmodels.tsa.stattools import adfuller

from xgboost import XGBClassifier

from sklearn.feature_selection import mutual_info_classif, RFECV
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, kendalltau
#


#
data = pandas.read_excel('./data/dataset_refined.xlsx')
data['datetime'] = pandas.to_datetime(data['datetime'])
data = data.set_index('datetime')


def lag(data, n_lags):
    data_lagged = data.copy()
    for j in range(n_lags):
        data_lagged[[x + '_LAG{0}'.format(j + 1) for x in data.columns.values]] = data.shift(periods=(j + 1))
    return data_lagged


data = lag(data=data, n_lags=20)
data = data.dropna()

# removables = []
removables = ['FLAG']

targets = ['_TARGET', 'TARGET_1M', 'TARGET_3M', 'TARGET_6M']
x_factors = [x for x in data.columns if not any([y in x for y in targets + removables])]

X = data[x_factors].values.astype(dtype=float)
Y = (data[targets[1]] > data[targets[0]]).values.astype(dtype=int)

ix_start = pandas.to_datetime('2006-12-27')
ix_thresh = pandas.to_datetime('2014-07-25')
ix_end = pandas.to_datetime('2021-06-27')

data = data[(data.index >= ix_start) * (data.index <= ix_end)]
thresh_val = numpy.where(data.index.values == ix_thresh)[0][0]

X_train, X_test = X[:thresh_val], X[thresh_val:]
Y_train, Y_test = Y[:thresh_val], Y[thresh_val:]


thresh = 0.01
# disc = [any([y in x for y in ['QU', 'QD', 'CFL', 'CFY']]) for x in x_factors]
values = mutual_info_classif(X=X_train, y=Y_train, discrete_features='auto')
fs_mask = values >= thresh

"""
thresh = 0.05
values = numpy.array([spearmanr(a=X_train[:, j], b=Y_train)[0] for j in range(X_train.shape[1])])
fs_mask = numpy.abs(values) >= thresh
"""
"""
thresh = 0.05
values = numpy.array([kendalltau(x=X_train[:, j], y=Y_train)[0] for j in range(X_train.shape[1])])
fs_mask = numpy.abs(values) >= thresh
"""
"""
alpha = 0.05
values = numpy.array([spearmanr(a=X_train[:, j], b=Y_train)[1] for j in range(X_train.shape[1])])
fs_mask = values <= alpha
"""
"""
alpha = 0.05
values = numpy.array([kendalltau(x=X_train[:, j], y=Y_train)[1] for j in range(X_train.shape[1])])
fs_mask = values <= alpha
"""

X_train = X_train[:, fs_mask]
X_test = X_test[:, fs_mask]


random.seed(999)
numpy.random.seed(999)
torch.manual_seed(999)
rs = 999

mkw = {'n_estimators': 1000, 'max_depth': None, 'min_samples_leaf': 1, 'random_state': rs}

# mkw = {'n_estimators': 1000, 'max_depth': 3, 'min_samples_leaf': 1, 'random_state': rs}
# mkw = {'n_estimators': 100, 'max_depth': 3, 'min_samples_leaf': 1, 'random_state': rs}
# mkw = {'n_estimators': 100, 'max_depth': None, 'min_samples_leaf': 1, 'random_state': rs}

# mkw = {'n_estimators': 1000, 'max_depth': None, 'use_label_encoder': False, 'random_state': rs}

# mkw = {'max_iter': 10_000}
# mkw = {'n_neighbors': 5, 'weights': 'uniform'}
# mkw = {'n_neighbors': 15, 'weights': 'uniform'}
# mkw = {'n_neighbors': 25, 'weights': 'distance'}
# mkw = {'n_neighbors': 5, 'weights': 'distance'}
# mkw = {'n_neighbors': 15, 'weights': 'distance'}
# mkw = {'n_neighbors': 25, 'weights': 'distance'}

# model = RandomForestClassifier(**mkw)
# model = ExtraTreesClassifier(**mkw)
# model = XGBClassifier(**mkw)
# model = LogisticRegression(**mkw)
# model = KNeighborsClassifier(**mkw)
model = RFECV(estimator=RandomForestClassifier(**mkw), n_jobs=-1)
model.fit(X=X_train, y=Y_train)

y_hat_train = model.predict(X=X_train)
y_hat_test = model.predict(X=X_test)

"""
Y_train_ = numpy.concatenate(((Y_train.reshape(-1, 1) == 0).astype(dtype=int),
                             (Y_train.reshape(-1, 1) == 1).astype(dtype=int)),
                            axis=1)
Y_test_ = numpy.concatenate(((Y_test.reshape(-1, 1) == 0).astype(dtype=int),
                            (Y_test.reshape(-1, 1) == 1).astype(dtype=int)),
                           axis=1)

X_train_ = torch.tensor(X_train, dtype=torch.float)
Y_train_ = torch.tensor(Y_train_, dtype=torch.float)
X_test_ = torch.tensor(X_test, dtype=torch.float)
Y_test_ = torch.tensor(Y_test_, dtype=torch.float)
"""
"""
nn_kwargs = {'layers': [nn.Linear, nn.Linear], # , nn.Softmax],
         'layers_dimensions': [10, 1], #, 1],
         'layers_kwargs': [{}, {}], # , {}],
         'activators': [None, nn.LeakyReLU], # , None],
         'interdrops': [0.1, 0.0], # , 0.0],
             'optimiser': torch.optim.Adamax,
             'optimiser_kwargs': {'lr': 0.002},
             'loss_function': nn.BCEWithLogitsLoss,
             'epochs': 2000
        #  'device': device,
             }
"""
"""
nn_kwargs = {'layers': [nn.Linear, nn.Linear, nn.Linear],
         'layers_dimensions': [32, 16, 2],
         'layers_kwargs': [{}, {}, {}],
         'activators': [None, nn.LeakyReLU, nn.Softmax],
         'interdrops': [0.3, 0.3, 0.0],
             'optimiser': torch.optim.Adamax,
             'optimiser_kwargs': {'lr': 0.003},
             'loss_function': nn.CrossEntropyLoss,
             'epochs': 2000
        #  'device': device,
             }
"""
"""
model = WrappedNN(**nn_kwargs)

model.fit(X_train=X_train_, Y_train=Y_train_, X_val=X_test_, Y_val=Y_test_)

y_hat_train = model.predict(X=X_train_)
y_hat_test = model.predict(X=X_test_)

y_hat_train = y_hat_train[:, 0] < y_hat_train[:, 1]
y_hat_test = y_hat_test[:, 0] < y_hat_test[:, 1]
"""


cm_train = confusion_matrix(y_true=Y_train, y_pred=y_hat_train)
cm_test = confusion_matrix(y_true=Y_test, y_pred=y_hat_test)

ac_train = accuracy_score(y_true=Y_train, y_pred=y_hat_train)
ac_test = accuracy_score(y_true=Y_test, y_pred=y_hat_test)

matt_train = matthews_corrcoef(y_true=Y_train, y_pred=y_hat_train)
matt_test = matthews_corrcoef(y_true=Y_test, y_pred=y_hat_test)

print(cm_train, '\n', cm_test, '\n',
      '{0:.4f}'.format(ac_train), '\n', '{0:.4f}'.format(matt_train), '\n',
      '{0:.4f}'.format(ac_test), '\n', '{0:.4f}'.format(matt_test), '\n')


def cc(mx):
    return (mx[1, 1] + mx[2, 2]) / (mx[1, 1] + mx[1, 2] + mx[2, 1] + mx[2, 2])

