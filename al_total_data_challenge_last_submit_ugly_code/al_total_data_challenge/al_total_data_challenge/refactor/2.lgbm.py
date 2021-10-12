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

# pretty_x_columns = sorted(list(data.select_dtypes(include='category').columns)) + sorted(list(data.select_dtypes(exclude='category').columns))


base_features = [
     'ws',
     'u',
     'v',
     ]
derivative_names = [
    # '_alphaOLS',
    '_betaOLS',
    '_r2OLS',
    '_d1_cl',
    # '_d1_ct',
    # '_d1_ll',
    # '_d1_rl',
    '_d2_cl',
    # '_d2_ct',
    '_d3_cl',
    # '_interp',
    ]

sliding_names = [
    '_mean_w',
    '_min_w',
    '_max_w',
    '_std_w',
    ]
sliding_lengths = [
    '3',
    # '5',
    '7',
    # '9',
    '11',
    # '13',
    # '15',
    # '19',
    ]


weird_derivatives = [
        # 'd2uv_cross_uvmean',
    'd2uv_cross_uvmean_n',
        # 'd2uv_dot___uvmean',
        # 'd2uv_dot___uvmean_n',
    'd2uv_dot___uvmean_n_abs',
    'd2uv_norm',
    'd3uv_norm',
        # 'duv_cross_uvmean',
    'duv_cross_uvmean_n',
        # 'duv_dot___uvmean',
    'duv_dot___uvmean_n',
    'duv_dot___uvmean_n_abs',
    'duv_norm',
    ]

neighbors_features = [
    'u_nei_lr',
    'v_nei_lr',
    'ws_nei_lr',
    'ws_nei_lr_calc',
    'ws_lr',
    'ws_lr2',
    'ws_mixed',
    ]

categoricals = [
    'am_pm',
    # 'f_for_12h_group',
    # 'forecast_from_hour',
    'hh',
    # 'hh2',
    # 'hh3',
    # 'hh4',
    # 'hh6',
    'mm',
    # 'mm2',
    # 'mm2hh2',
    'mm2hh3',
    # 'mm3',
    # 'mm3hh4',
    # 'mm4',
    # 'mmhh',
    # 'mmhh2',
    # 'mmhh3',
    # 'yy',
    'yymm4',
    ]

misc = [
    'farm_number',
    # 'forecast_for',
    # 'forecast_from',
    'horizon',
    # 'hors',
    # 'hour_in_b12',
    'hour_in_b48',
    # 'wp',
    ]

cluster_names = [
    # 'km_uv_block_',
    # 'km_ws_block_',
        # 'km_uv_farms_', 
    # 'km_ws_farms_', 
    ]

cluster_counts = [
    # '16',
        # '32',
    # '48', 
    # '64', 
    ]

feats = []
feats_com = []

feats += base_features + ['wd']
feats += [base_feature + derivative_name for base_feature in base_features for derivative_name in derivative_names]
feats += [base_feature + name + length for base_feature in base_features for name in sliding_names for length in sliding_lengths]
feats += weird_derivatives
feats += neighbors_features
feats += [name + length for length in sliding_lengths for name in ['wsuv_w', 'wd_mean_w', 'uv_mean_w', 'uv_std_w']]

feats_com += categoricals
feats_com += misc
feats_com += [name + count for name in cluster_names for count in cluster_counts]

# sorted(list(set(list(data.columns)) - set(feats)))

data['is_test'] = data['wp'].isna()
data = data.set_index(['forecast_for', 'forecast_from', 'is_test']).sort_index()
data = data.reset_index()
#%%

models = [
{
    'name'      : 'a.lgbm',
    'horizons'  : [slice(None)],
    'farms'     : [slice(None)],
    'hh4ofdays' : [slice(None)],
    'lgbm_param':   {
                        'n_estimators' : 1800,
                        # 'num_leaves':50,
                        'learning_rate' : 0.02,
                        'objective':'mape',
                        'colsample_bytree':0.8,
                        # 'bagging_fraction':0.8,
                        # 'min_data_in_leaf':20,
                        'num_leaves':50,
                        }
},
{
    'name'      : 'c.lgbm',
    'horizons'  : list(range(0, 4)),
    'farms'     : [slice(None)],
    'hh4ofdays' : [slice(None)],
    'lgbm_param':   {
                        'n_estimators' : 1200,
                        # 'num_leaves':50,
                        'learning_rate' : 0.02,
                        'objective':'mape',
                        'colsample_bytree':0.8,
                        # 'bagging_fraction':0.8,
                        # 'min_data_in_leaf':20,
                        'num_leaves':50,
                        }
},
{
    'name'      : 'd.lgbm',
    'horizons'  : [slice(None)],
    'farms'     : [slice(None)],
    'hh4ofdays' : list(range(0, 6)),
    'lgbm_param':   {
                        'n_estimators' : 1200,
                        # 'num_leaves':50,
                        'learning_rate' : 0.02,
                        'objective':'mape',
                        'colsample_bytree':0.8,
                        # 'bagging_fraction':0.8,
                        # 'min_data_in_leaf':20,
                        'num_leaves':50,
                        }
},

# {
#     'name'      : 'c.lgbm',
#     'horizons'  : list(range(0, 4)),
#     'farms'     : [slice(None)],
#     'hh4ofdays' : [slice(None)],
#     'lgbm_param':   {
#                         'n_estimators' : 1200,
#                         # 'num_leaves':50,
#                         'learning_rate' : 0.15,
#                         'objective':'mae',
#                         'colsample_bytree':0.8,
#                         # 'bagging_fraction':0.8,
#                         # 'min_data_in_leaf':20,
#                         'num_leaves':100,
#                         }
# },
# {
#     'name'      : 'd.lgbm',
#     'horizons'  : [slice(None)],
#     'farms'     : [slice(None)],
#     'hh4ofdays' : list(range(0, 6)),
#     'lgbm_param':   {
#                         'n_estimators' : 1200,
#                         # 'num_leaves':50,
#                         'learning_rate' : 0.15,
#                         'objective':'mae',
#                         'colsample_bytree':0.8,
#                         # 'bagging_fraction':0.8,
#                         'min_data_in_leaf':20,
#                         'num_leaves':100,
#                         }
# },
]

features = feats + feats_com

import lightgbm as lgb

from package.ts_split_predict import ts_split_predict_lgbm
for model in models:
    pred = ts_split_predict_lgbm(data, split_dates_start, split_dates_end, features, model)
    pd.DataFrame(pred).reset_index(level=-1, drop=True).to_parquet('./model_predictions/'+model['name']+'.parquet.gzip',
                                    engine = 'fastparquet',
                                    compression='gzip')

