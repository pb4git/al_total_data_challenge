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
    # 'farm_number',
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
# {
#     'name'      : 'a.all',
#     'horizons'  : [slice(None)],
#     'hh4ofdays' : [slice(None)],
#     'catboost_param':   {
#                         'n_estimators' : 1800,
#                         'bootstrap_type':'Bernoulli',
#                         'loss_function':'MAPE',
#                         'learning_rate' : 0.016,
#                         'depth':8,
#                         'l2_leaf_reg':0.08,
#                         'colsample_bylevel':0.75,
#                         'random_strength':0.4,
#                         'subsample':0.65,
#                         },
#     'farms_to_use_as_features' : {1:[1,2,3,4,5,6],
#                                   2:[1,2,3,4,5,6],
#                                   3:[1,2,3,4,5,6],
#                                   4:[1,2,3,4,5,6],
#                                   5:[1,2,3,4,5,6],
#                                   6:[1,2,3,4,5,6]}
# },
# {
#     'name'      : 'c.all',
#     'horizons'  : list(range(0, 4)),
#     'hh4ofdays' : [slice(None)],
#     'catboost_param':   {
#                         'n_estimators' : 1200,
#                         'bootstrap_type':'Bernoulli',
#                         'loss_function':'MAPE',
#                         'learning_rate' : 0.016,
#                         'depth':8,
#                         'l2_leaf_reg':0.08,
#                         'colsample_bylevel':0.75,
#                         'random_strength':0.4,
#                         'subsample':0.65,
#                         },
#     'farms_to_use_as_features' : {1:[1,2,3,4,5,6],
#                                   2:[1,2,3,4,5,6],
#                                   3:[1,2,3,4,5,6],
#                                   4:[1,2,3,4,5,6],
#                                   5:[1,2,3,4,5,6],
#                                   6:[1,2,3,4,5,6]}
# },
{
    'name'      : 'd.all',
    'horizons'  : [slice(None)],
    'hh4ofdays' : list(range(0, 6)),
    'catboost_param':   {
                        'n_estimators' : 1200,
                        'bootstrap_type':'Bernoulli',
                        'loss_function':'MAPE',
                        'learning_rate' : 0.016,
                        'depth':8,
                        'l2_leaf_reg':0.08,
                        'colsample_bylevel':0.75,
                        'random_strength':0.4,
                        'subsample':0.65,
                        },
    'farms_to_use_as_features' : {1:[1,2,3,4,5,6],
                                  2:[1,2,3,4,5,6],
                                  3:[1,2,3,4,5,6],
                                  4:[1,2,3,4,5,6],
                                  5:[1,2,3,4,5,6],
                                  6:[1,2,3,4,5,6]}
},
]

features = feats + feats_com

from package.ts_split_predict import ts_split_predict_2
for model in models:
    pred = ts_split_predict_2(data, split_dates_start, split_dates_end, feats, feats_com, model)
    pd.DataFrame(pred).reset_index(level=-1, drop=True).to_parquet('./model_predictions/'+model['name']+'.parquet.gzip',
                                    engine = 'fastparquet',
                                    compression='gzip')

qqqqqqqqqqqq

##################################
    

a = data.pivot(index=['forecast_for', 'forecast_from', 'is_test'], columns='farm_number')
b = pd.concat([a[features], a[list(zip(features_com, repeat(1)))].droplevel(1, axis=1)], axis=1)
c = a['wp']
#%%

for farm_number in range(1, 7):
    for s, e in zip(xv_start, xv_end):
    
        data = data.reset_index()
        data['yy'] = ((s-data['forecast_for'])//pd.to_timedelta('8760h')).clip(lower=0)
        data['yymm4'] = ((s-data['forecast_for'])//pd.to_timedelta('2920h')).clip(lower=0)
        data = data.set_index(['forecast_for', 'forecast_from', 'is_test']).sort_index()
    
        
        X = b.loc[(slice(s - pd.to_timedelta('1h')), slice(None), False),:]
        y = c.loc[(slice(s - pd.to_timedelta('1h')), slice(None), False),farm_number]
    
        X_p = b.loc[(slice(s, e), slice(None), slice(None)),:]
        y_p = c.loc[(slice(s, e), slice(None), slice(None)),farm_number]
        
        print(X.shape, y.shape, X_p.shape)
        cb = SqrtSquareCBRegressor(catboost.CatBoostRegressor(n_estimators=1000, bootstrap_type='Bernoulli', subsample=0.8, verbose=1, loss_function='MAE'))
    
        cf = list(X.select_dtypes(include='category').columns)
        cb.fit(X, y, cat_features=cf)
        
        b.loc[(slice(s, e), slice(None), slice(None)),'pred'+str(farm_number)] = cb.predict(X_p)
#        print(mean_absolute_error(data['pred'][~(data[['pred', 'wp']].isna().any(axis=1))], data['wp'][~(data[['pred', 'wp']].isna().any(axis=1))]))


#%%
wp2 = pd.concat([a[['wp']], b['horizon']], axis=1).set_index('horizon', append=True)
pred2 = b[['pred'+str(i) for i in range(1, 7)]+['horizon']].set_index('horizon', append=True)

wp2.columns=list(range(1,7))
pred2.columns=list(range(1,7))

wp2 = wp2.stack()
pred2 = pred2.stack()

comb = pd.concat([wp2, pred2], axis=1)
comb.columns = ['wp', 'pred']
comb = comb.reset_index(level=-1).rename(columns={'level_4':'farm_number'}).set_index('farm_number',append=True)
comb = comb.reset_index()
comb = comb.set_index(['forecast_for', 'farm_number', 'horizon', 'forecast_from', 'is_test']).sort_index()
comb2 = comb.sort_index().groupby(level=[0, 1]).first()

print(mean_absolute_error(comb2[~comb2.isna().any(axis=1)]['wp'], comb2[~comb2.isna().any(axis=1)]['pred']))
#%%
        
        
#data = data.reset_index()
#data = data.set_index(['forecast_for', 'horizon', 'farm_number']).sort_index()
#a = data[['pred', 'wp']].unstack().sort_index().groupby(level=0).first()    
##a = data[['pred', 'wp']].unstack().sort_index()
#print(mean_absolute_error(a[~(a.isna().any(axis=1))].stack()['pred'], a[~(a.isna().any(axis=1))].stack()['wp'])) #for score saving
#


#predictions = test_dates.copy()
#predictions.index = predictions.index.rename('forecast_for')
#predictions = predictions.join(other=a['pred'])
#predictions.index = predictions.index.rename('date')
#predictions.index = predictions.index.strftime(date_format='%Y%m%d%H')
#predictions = predictions.stack().reset_index(level=1).rename(columns={'level_1':'farm_number', 0:'wppred'}).set_index('farm_number', append=True)
#predictions = predictions.unstack()


predictions = test_dates.copy()
predictions.index = predictions.index.rename('forecast_for')
predictions = predictions.join(other=comb2['pred'].unstack(level=1))
predictions.index = predictions.index.rename('date')
predictions.index = predictions.index.strftime(date_format='%Y%m%d%H')
predictions = predictions.stack().reset_index(level=1).rename(columns={'level_1':'farm_number', 0:'wppred'}).set_index('farm_number', append=True)
predictions = predictions.unstack()
#%%
predictions.columns = ['wp'+str(i) for i in range(1, 7)]

#%%
pd.DataFrame({'name': cb.regr.feature_names_, 'importance':cb.regr.feature_importances_}).sort_values('importance')

# %%
# ========================================================================================
#   CREATE SUBMISSION DATAFILE
# ========================================================================================
# DONTCREATESUBMISSION

now = datetime.datetime.now()
tag = now.strftime("%m-%d_%H:%M:%S")


def copy_rename(old_file_name, new_file_name):
    src_dir = os.curdir
    print(src_dir)
    dst_dir = os.path.join(os.curdir, "../saved_codes")
    src_file = os.path.join(src_dir, old_file_name)
    print(src_file)
    shutil.copy(src_file, dst_dir)

    dst_file = os.path.join(dst_dir, old_file_name)
    new_dst_file_name = os.path.join(dst_dir, new_file_name)
    os.rename(dst_file, new_dst_file_name)

predictions.to_csv('/home/paul/Documents/ALTotal Challenge/submissions/'+tag+'.csv', sep=';')
copy_rename(FILENAME, tag+'.py')

#%%
    
    


aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
data = farm.join(other=wp)

f_ws = pd.DataFrame()
f_wp = pd.DataFrame()

for i in range(0, 6):
    data = farms[i].join(other=wp)
    #sns.scatterplot(x=data.loc[(data['hors'] < 13) & (~data['ws'].isna()), 'ws'], y=data.loc[(data['hors'] < 13) & (~data['ws'].isna()), 'wp'+str(1+i)], s=2)
    #plt.show()
    
    f_ws[i] = data.loc[(data['hors'] < 13) & (~data['ws'].isna()), 'ws'].copy()
    f_wp[i] = data.loc[(data['hors'] < 13) & (~data['ws'].isna()), 'wp'+str(1+i)]
    
ica = FastICA()
f_ws_ica = ica.fit_transform(f_ws)
f_ws_ica = pd.DataFrame(f_ws_ica, index=f_ws.index)

pca = PCA()
f_ws_pca = pca.fit_transform(f_ws)
f_ws_pca = pd.DataFrame(f_ws_pca, index=f_ws.index)

from sklearn.cross_decomposition import PLSRegression
pls = PLSRegression(n_components=6)
pls.fit(f_ws[~f_wp.isna().any(axis=1)], f_wp[~f_wp.isna().any(axis=1)])
f_ws_pls = pls.transform(f_ws)
f_ws_pls = pd.DataFrame(f_ws_pls, index=f_ws.index)


sns.heatmap(f_ws_pls.join(other=f_ws_pca.join(other=f_ws_ica.join(other=f_ws.join(other=f_wp, lsuffix='_ws', rsuffix='_wp'), lsuffix='ica'), lsuffix='pca'),lsuffix='pls').corr().abs())

pca = PCA()

aaaaaaaa
data.date = pd.to_datetime(data.date, format='%Y-%m-%d %H:%M:%S')
data.set_index('date', inplace=True)

test = pd.read_csv(f'{data_path}test_phase_1.csv')
test.date = pd.to_datetime(test.date, format='%Y-%m-%d %H:%M:%S')
test.set_index('date', inplace=True)


data = data.append(test)
data = data.sort_index()

#%%

for i in range(10, 20):
    sns.lineplot(data=a.iloc[:,i])
    sns.lineplot(data=r.iloc[:,i])
    sns.lineplot(data=l.iloc[:,i])
    plt.show()
    