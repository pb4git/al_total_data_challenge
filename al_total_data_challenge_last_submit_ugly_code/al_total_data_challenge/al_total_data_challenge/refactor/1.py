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

import package.featureengineering
data = package.featureengineering.featureengineering(data) 

data.to_parquet('data.1.1.0.0.parquet.gzip',
                engine = 'fastparquet',
                compression='gzip')

#%%