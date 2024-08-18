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


# data = lag(data=data, n_lags=20)
data = data.dropna()

removables = []
# removables = ['FLAG']

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

