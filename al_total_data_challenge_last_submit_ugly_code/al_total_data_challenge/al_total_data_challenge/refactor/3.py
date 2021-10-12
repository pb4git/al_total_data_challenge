# %%#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: paul
"""
import shutil
import datetime
# import time
# import math
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Input, Dense, Flatten, Activation, LeakyReLU, Conv2D, LSTM, Dropout, Flatten, Conv1D
# from tensorflow.keras.models import Model
# from tensorflow.keras.utils import plot_model
# from tensorflow.keras.optimizers import Adam, SGD, Nadam
# from tensorflow.keras.regularizers import l2
# import tensorflow as tf
# import hyperopt
# from xgboost import XGBRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import cross_val_predict
# from sklearn.svm import SVR
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.kernel_ridge import KernelRidge
# from sklearn.base import BaseEstimator
# from sklearn.cluster import KMeans
# from sklearn.linear_model import RidgeCV, Ridge
# import catboost

from pandas.api.types import CategoricalDtype

import os
# import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# from scipy.interpolate import interp1d
# from scipy import interpolate
# import scipy
# import itertools

# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, r'C:\Users\paul.berhaut\projects\al_total_data_challenge\data\challenge_19_data\package\\')

# ========================================================================================
# IMPORTS
# ========================================================================================

pd.set_option('display.max_rows', 4000)
pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 1000)
pd.options.display.max_colwidth = 100

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
# tf.random.set_seed(RANDOM_SEED)
# tf.compat.v1.disable_eager_execution()

# %%

fig_dir = r'C:\Users\paul.berhaut\projects\al_total_data_challenge\figures'
for f in os.listdir(fig_dir):
    os.remove(os.path.join(fig_dir, f))
fig_number = 0

############

import package.readrawdata
data, test_dates = package.readrawdata.readrawdata(r'C:\Users\paul.berhaut\projects\al_total_data_challenge\data\challenge_19_data\\')

data = pd.read_parquet('data.1.0.0.0.parquet.gzip',
                       engine='fastparquet') #fastparquet required to preserve categorical data types

import package.split
split_dates_start, split_dates_end = package.split.get_xval_split_boundaries(data, 52)

split_dates_start.insert(0, pd.to_datetime('2010-03-01 01:00:00'))
split_dates_end.insert(0, pd.to_datetime('2011-01-01 00:00:00'))

#%%%%%%%%%

data['is_test'] = data['wp'].isna()
data = data.set_index(['forecast_for', 'forecast_from', 'is_test']).sort_index()
data = data.reset_index()
#%%

data = data.set_index(['forecast_for', 'farm_number', 'horizon', 'forecast_from'])

# model_names = ['a', 'b']
model_names = [
    'a', 
    # 'b', 
    # 'c', 
    # 'd', 
    # 'a.all', 
    'c.all',
    # 'a.some.one', 
    'a.some.three', 
    # 'a.quant', 
    # 'b.quant', 
    # 'c.quant', 
    # 'd.quant', 
    # 'a.power',
    # 'b.power', 
    # 'c.power', 
    # 'd.power',
    # 'a.otherfeatures',
    # 'c.otherfeatures',
    # 'd.otherfeatures',
    'a.all.hordelta',
            # 'a.lgbm',
            # 'c.lgbm',
            # 'd.lgbm',
            # 'a5.lgbm',
            # 'a6.lgbm',
            # 'a7.lgbm',
            # 'a8.lgbm',
            # 'a9.lgbm',
            # 'a10.lgbm',
            # 'a11.lgbm',
            # 'a12.lgbm',
            # 'a13.lgbm',
            # 'a14.lgbm',
            # 'a15.lgbm',
    # 'a.lgbm',
    # 'c.lgbm',
    # 'd.lgbm',
    # 'a.all.lgbm',
    'c.all.lgbm',
    ]
model_names_ae = [model_name + '_ae' for model_name in model_names]

for model_name in model_names:
    print('read', model_name)
    data[model_name] = pd.read_parquet(r'.\model_predictions\\'+model_name+'.parquet.gzip',engine='fastparquet')
    data[model_name+'_ae'] = (data[model_name] - data['wp']).abs()
        
data = data.reset_index()

#%%
data = data.set_index(['forecast_for', 'horizon', 'farm_number'])
a = data.unstack()['a']

data['roll'] = a.ewm(span=1.5).mean().stack()
data = data.reset_index()


#%%
from itertools import repeat
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator
import numpy as np
import catboost

# class OneRidge(BaseEstimator):
#     def __init__(self):
#         self.regr = Ridge(fit_intercept=False)

#     def fit(self, X, y):
#         self.regr.fit(X, y)
#         self.regr.coef_ /= np.sum(self.regr.coef_)
#         return self

#     def predict(self, X):
#         return self.regr.predict(X)

class OneRidge(Ridge):
    def fit(self, X, y):
        super().fit(X, y)
        self.coef_ /= np.sum(self.coef_)
        return self



for base_model_name in model_names:
    print('Horiz', base_model_name)

    data = data.set_index(['forecast_for', 'farm_number', 'horizon']).sort_index()
    a = data[[base_model_name, 'wp']].unstack(level=2)
    b = pd.DataFrame(index=a.index)

    for n_predictions in range(4, 0, -1):
        
        predictions_available = list(range(4-n_predictions, 4))
            
        mask = ~(a[zip(repeat(base_model_name), predictions_available)].isna().any(axis=1))
        mask2 = ~(a['wp'].isna().any(axis=1))
        
        mask_a_and_wp = mask & mask2
        mask_a_and_not_wp = mask & ~mask2
        
        mask_date = a.index.get_level_values(0).year <= 2010
        
        X = a[mask_a_and_wp & mask_date][zip(repeat(base_model_name), predictions_available)]
        y = a[mask_a_and_wp & mask_date][('wp', 3)]

        # if n_predictions > 2:
        #     X['std'] = X.std(axis=1)

        lr = Ridge(fit_intercept=False) #998
        # lr = OneRidge() #1007
        # lr = catboost.CatBoostRegressor(loss_function='MAE') #996

        lr.fit(X, y)
        # print(n_predictions, lr.coef_)
        b.loc[mask_a_and_not_wp | mask_a_and_wp, 4-n_predictions] = lr.predict(a[mask_a_and_not_wp | mask_a_and_wp][zip(repeat(base_model_name), predictions_available)])
        
    b.columns = pd.MultiIndex.from_product([['lrhoriz_'+base_model_name], list(range(0, 4))], names=(None, 'farm_number'))
    b['lrhoriz_'+base_model_name] = b['lrhoriz_'+base_model_name].fillna(value=a[base_model_name])
    data['lrhoriz_'+base_model_name] = b.stack()

    data = data.reset_index()

#%%

data['fn_horizon_hh4'] =             pd.Categorical( data['farm_number'].astype(int)*4*6 +  data['horizon'].astype(int)*6 +  data['hh4'].astype(int))
data['fn_horizon']     =             pd.Categorical( data['farm_number'].astype(int)*4   +  data['horizon'].astype(int))

data['fn_horizon']     =             pd.Categorical( data['farm_number'].astype(int)*0 + data['horizon'].astype(int)*0)

c = pd.concat([pd.get_dummies(data['fn_horizon']).mul(data[model_name], axis=0) for model_name in model_names], axis=1)
c = pd.concat([c, data['wp']], axis=1)

c.columns = list(range(len(c.columns)-1)) + ['wp']

lrfit = (~c.isna().any(axis=1)) & (data["forecast_for"].dt.year <= 2010)
lruse = ~(c.drop('wp', axis=1).isna().any(axis=1))

from sklearn.linear_model import Ridge, RidgeCV, Lasso
lr = Ridge(alpha=1, fit_intercept=False)
# from sklearn.neural_network import MLPRegressor
# lr = MLPRegressor(hidden_layer_sizes=(10, 10, 10), activation='relu')
lr = catboost.CatBoostRegressor(loss_function='MAE')

# lr = RidgeCV(alphas=[0.1, 0.3, 1, 3, 10, 30, 100], fit_intercept=False)

lr.fit(c[lrfit].drop("wp", axis=1), c[lrfit]["wp"])
data.loc[lruse, 'lr'] = lr.predict(c[lruse].drop("wp", axis=1))


#%%

data['fn_horizon_hh4'] =             pd.Categorical( data['farm_number'].astype(int)*4*6 +  data['horizon'].astype(int)*6 +  data['hh4'].astype(int))
data['fn_horizon']     =             pd.Categorical( data['farm_number'].astype(int)*4   +  data['horizon'].astype(int))

data['fn_horizon']     =             pd.Categorical( data['farm_number'].astype(int)*0 + data['horizon'].astype(int)*0)


c = pd.concat([pd.get_dummies(data['fn_horizon']).mul(data['lrhoriz_'+model_name], axis=0) for model_name in model_names], axis=1)
c = pd.concat([c, data['wp']], axis=1)

lrfit = (~c.isna().any(axis=1)) & (data["forecast_for"].dt.year <= 2010)
lruse = ~(c.drop('wp', axis=1).isna().any(axis=1))

from sklearn.linear_model import Ridge, RidgeCV, Lasso
lr = OneRidge(alpha=1, fit_intercept=False)
# from sklearn.neural_network import MLPRegressor
# lr = MLPRegressor(hidden_layer_sizes=(5, 5), activation='relu')
# lr = RidgeCV(alphas=[0.1, 0.3, 1, 3, 10, 30, 100], fit_intercept=False)

lr.fit(c[lrfit].drop("wp", axis=1), c[lrfit]["wp"])
data.loc[lruse, 'lrhoriz_lr'] = lr.predict(c[lruse].drop("wp", axis=1))


#%%

data['fn_horizon_hh4'] =             pd.Categorical( data['farm_number'].astype(int)*4*6 +  data['horizon'].astype(int)*6 +  data['hh4'].astype(int))
data['fn_horizon']     =             pd.Categorical( data['farm_number'].astype(int)*4   +  data['horizon'].astype(int))

data['fn_horizon']     =             pd.Categorical( data['farm_number'].astype(int)*0 + data['horizon'].astype(int)*0)


c = pd.concat([pd.get_dummies(data['fn_horizon']).mul(data['lrhoriz_'+model_name], axis=0) for model_name in model_names]+[pd.get_dummies(data['fn_horizon']).mul(data[model_name], axis=0) for model_name in model_names], axis=1)
c = pd.concat([c, data['wp']], axis=1)

lrfit = (~c.isna().any(axis=1)) & (data["forecast_for"].dt.year <= 2010)
lruse = ~(c.drop('wp', axis=1).isna().any(axis=1))

from sklearn.linear_model import Ridge, RidgeCV, Lasso
lr = OneRidge(alpha=1, fit_intercept=False)
# lr = RidgeCV(alphas=[0.1, 0.3, 1, 3, 10, 30, 100], fit_intercept=False)

lr.fit(c[lrfit].drop("wp", axis=1), c[lrfit]["wp"])
data.loc[lruse, 'lrhoriz2_lr'] = lr.predict(c[lruse].drop("wp", axis=1))


#%%
# from sklearn.model_selection import cross_val_predict
# from sklearn.ensemble import RandomForestRegressor
# from catboost import CatBoostRegressor

# rfr = CatBoostRegressor(cat_features=['farm_number', 'horizon', 'hh'])

# features = model_names + ['farm_number', 'horizon', 'hh']
# a = data[features + ['wp']]

# fitmask = ~a.isna().any(axis=1)
# usemask = ~(a.drop('wp', axis=1).isna().any(axis=1))

# data.loc[fitmask, 'rfr'] = cross_val_predict(rfr, a[fitmask].drop('wp', axis=1), a[fitmask]['wp'])

# rfr.fit(a[fitmask].drop('wp', axis=1), a[fitmask]['wp'])
# data.loc[usemask & ~fitmask, 'rfr'] = rfr.predict(a[usemask & ~fitmask].drop('wp', axis=1))

#%%
# from sklearn.model_selection import cross_val_predict
# from catboost import CatBoostClassifier

# rfc = CatBoostClassifier(cat_features=['farm_number', 'horizon', 'hh'], n_estimators=200)

# features = model_names + ['farm_number', 'horizon', 'hh']
# a = data[features]
# a = pd.concat([a, data[model_names_ae].idxmin(axis=1)], axis=1)

# fitmask = ~a.isna().any(axis=1)
# usemask = ~(a.drop(0, axis=1).isna().any(axis=1))

# t = cross_val_predict(rfc, a[fitmask].drop(0, axis=1), a[fitmask][0], method='predict_proba')
# data.loc[fitmask, [model_name + '_cw' for model_name in model_names]] = t

# rfc.fit(a[fitmask].drop(0, axis=1), a[fitmask][0])
# data.loc[usemask & ~fitmask, [model_name + '_cw' for model_name in model_names]] = rfc.predict_proba(a[usemask & ~fitmask].drop(0, axis=1))


# data['rfc'] = pd.concat([data[model_name].mul(data[model_name+'_cw'], axis=0) for model_name in model_names], axis=1).sum(axis=1)
#%%
to_add_to_model_names_ae = []
to_add_to_model_names = []

data['mean'] = data[model_names].mean(axis=1)
data['mean_ae'] = (data['mean'] - data['wp']).abs()

to_add_to_model_names.append('mean')
to_add_to_model_names_ae.append('mean_ae')
#%%
# for model_name in model_names:
#     data[model_name + '_lrfarms_ae'] = (data[model_name + '_lrfarms'] - data['wp']).abs()
# model_names_ae += [model_name + '_lrfarms_ae' for model_name in model_names]
# model_names += [model_name + '_lrfarms' for model_name in model_names]

#%%

for base_model_name in model_names:
    data['lrhoriz_'+base_model_name+'_ae'] = (data['lrhoriz_'+base_model_name] - data['wp']).abs()
    to_add_to_model_names_ae.append('lrhoriz_'+base_model_name+'_ae')
    to_add_to_model_names.append('lrhoriz_'+base_model_name)

#%%
data['lr_ae'] = (data['lr'] - data['wp']).abs()

to_add_to_model_names.append('lr')
to_add_to_model_names_ae.append('lr_ae')
#%%

data['lrhoriz_lr_ae'] = (data['lrhoriz_lr'] - data['wp']).abs()

to_add_to_model_names.append('lrhoriz_lr')
to_add_to_model_names_ae.append('lrhoriz_lr_ae')
#%%
data['lrhoriz2_lr_ae'] = (data['lrhoriz2_lr'] - data['wp']).abs()

to_add_to_model_names.append('lrhoriz2_lr')
to_add_to_model_names_ae.append('lrhoriz2_lr_ae')

#%%
data['roll_ae'] = (data['roll'] - data['wp']).abs()
to_add_to_model_names.append('roll')
to_add_to_model_names_ae.append('roll_ae')
#%%
# data['rfr_ae'] = (data['rfr'] - data['wp']).abs()
# model_names += ['rfr']
# model_names_ae += ['rfr_ae']
#%%
# data['rfc_ae'] = (data['rfc'] - data['wp']).abs()
# model_names += ['rfc']
# model_names_ae += ['rfc_ae']

model_names += to_add_to_model_names
model_names_ae += to_add_to_model_names_ae

#%%
data['horizon_copy'] = data['horizon']
a = data[['forecast_for', 'farm_number', 'horizon', 'horizon_copy', 'forecast_from']+model_names+model_names_ae].set_index(['forecast_for', 'farm_number','horizon_copy']).sort_index().groupby(level=[0, 1]).first()

#%%

b = a.loc[(slice(pd.to_datetime('2011-01-01 00:00:00'),pd.to_datetime('2015-01-01 00:00:00')), slice(None)),:]
b = b.set_index(['horizon', 'forecast_from'], append=True)
print(b[model_names_ae].groupby(level=1).mean())
print(b[model_names_ae].groupby(level=1).mean().mean())
sns.heatmap(b[model_names_ae].groupby(level=1).mean())

#%%
predictions = test_dates.copy()
predictions.index = predictions.index.rename('forecast_for')
predictions = predictions.join(other=b['lr'].reset_index().set_index(['forecast_for', 'farm_number'])['lr'].unstack())
predictions.index = predictions.index.rename('date')
predictions.index = predictions.index.strftime(date_format='%Y%m%d%H')
predictions = predictions.stack().reset_index(level=1).rename(columns={'level_1':'farm_number', 0:'wppred'}).set_index('farm_number', append=True)
predictions = predictions.unstack()
now = datetime.datetime.now()
tag = now.strftime("%m-%d_%H:%M:%S")
predictions.columns = ['wp'+str(i) for i in range(1, 7)]
predictions.to_csv(path_or_buf='submit.csv', sep=';')

