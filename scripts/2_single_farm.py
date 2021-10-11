# %%#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: paul
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from al_total_data_challenge.readrawdata import readrawdata
from al_total_data_challenge.split import get_timeseries_split_boundaries
from al_total_data_challenge.regressors import (
    SqrtSquareCBRegressor,
    SqrtSquareLGBMRegressor,
    PowerCBRegressor,
    QuantileCBRegressor,
)
import catboost
from al_total_data_challenge.ts_split_predict import ts_split_predict

pd.set_option("display.max_rows", 4000)
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 1000)
pd.options.display.max_colwidth = 100

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

_, test_dates = readrawdata(r"..\data\challenge_19_data\\")

data = pd.read_parquet(
    "..\data.1.1.0.0.parquet.gzip", engine="fastparquet"
)  # fastparquet required to preserve categorical data types

split_start_dates, split_end_dates = get_timeseries_split_boundaries(data, 52)

split_start_dates.insert(0, pd.to_datetime("2010-03-01 01:00:00"))
split_end_dates.insert(0, pd.to_datetime("2011-01-01 00:00:00"))

########################################################################################
# Below is a list of all features that have been precalculated previously
# Comment to disable the feature, uncomment to use the feature in training.

base_features = [
    "ws",
    "u",
    "v",
]
derivative_names = [
    # '_alphaOLS',
    "_betaOLS",
    "_r2OLS",
    "_d1_cl",
    # '_d1_ct',
    # '_d1_ll',
    # '_d1_rl',
    "_d2_cl",
    # '_d2_ct',
    "_d3_cl",
    # '_interp',
]

sliding_names = [
    "_mean_w",
    "_min_w",
    "_max_w",
    "_std_w",
]
sliding_lengths = [
    "3",
    # '5',
    "7",
    # '9',
    "11",
    # '13',
    # '15',
    # '19',
]


weird_derivatives = [
    # 'd2uv_cross_uvmean',
    "d2uv_cross_uvmean_n",
    # 'd2uv_dot___uvmean',
    # 'd2uv_dot___uvmean_n',
    "d2uv_dot___uvmean_n_abs",
    "d2uv_norm",
    "d3uv_norm",
    # 'duv_cross_uvmean',
    "duv_cross_uvmean_n",
    # 'duv_dot___uvmean',
    "duv_dot___uvmean_n",
    "duv_dot___uvmean_n_abs",
    "duv_norm",
]

neighbors_features = [
    "u_nei_lr",
    "v_nei_lr",
    "ws_nei_lr",
    "ws_nei_lr_calc",
    "ws_lr",
    "ws_lr2",
    "ws_mixed",
]

categoricals = [
    "am_pm",
    # 'f_for_12h_group',
    # 'forecast_from_hour',
    "hh",
    # 'hh2',
    # 'hh3',
    # 'hh4',
    # 'hh6',
    "mm",
    # 'mm2',
    # 'mm2hh2',
    "mm2hh3",
    # 'mm3',
    # 'mm3hh4',
    # 'mm4',
    # 'mmhh',
    # 'mmhh2',
    # 'mmhh3',
    # 'yy',
    "yymm4",
]

misc = [
    "farm_number",
    # 'forecast_for',
    # 'forecast_from',
    "horizon",
    # 'hors',
    # 'hour_in_b12',
    "hour_in_b48",
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

feats += base_features + ["wd"]
feats += [
    base_feature + derivative_name
    for base_feature in base_features
    for derivative_name in derivative_names
]
feats += [
    base_feature + name + length
    for base_feature in base_features
    for name in sliding_names
    for length in sliding_lengths
]
feats += weird_derivatives
feats += neighbors_features
feats += [
    name + length
    for length in sliding_lengths
    for name in ["wsuv_w", "wd_mean_w", "uv_mean_w", "uv_std_w"]
]

feats_com += categoricals
feats_com += misc
feats_com += [name + count for name in cluster_names for count in cluster_counts]

########################################################################################
data["is_test"] = data["wp"].isna()
data = data.set_index(["forecast_for", "forecast_from", "is_test"]).sort_index()
data = data.reset_index()
########################################################################################
# Main list of models that are trained

models = [
    {
        "name": "a",
        "horizons": [slice(None)],
        "farms": [slice(None)],
        "hh4ofdays": [slice(None)],
        "model": SqrtSquareCBRegressor(
            catboost.CatBoostRegressor(
                n_estimators=2000,
                bootstrap_type="Bernoulli",
                loss_function="MAPE",
                learning_rate=0.016,
                depth=8,
                l2_leaf_reg=0.08,
                colsample_bylevel=0.75,
                random_strength=0.4,
                subsample=0.65,
                verbose=0,
            )
        ),
    },
    {
        "name": "b",
        "horizons": [slice(None)],
        "farms": list(range(1, 7)),
        "hh4ofdays": [slice(None)],
        "model": SqrtSquareCBRegressor(
            catboost.CatBoostRegressor(
                n_estimators=1200,
                bootstrap_type="Bernoulli",
                loss_function="MAPE",
                learning_rate=0.016,
                depth=8,
                l2_leaf_reg=0.08,
                colsample_bylevel=0.75,
                random_strength=0.4,
                subsample=0.65,
                verbose=0,
            )
        ),
    },
    {
        "name": "c",
        "horizons": list(range(0, 4)),
        "farms": [slice(None)],
        "hh4ofdays": [slice(None)],
        "model": SqrtSquareCBRegressor(
            catboost.CatBoostRegressor(
                n_estimators=1200,
                bootstrap_type="Bernoulli",
                loss_function="MAPE",
                learning_rate=0.016,
                depth=8,
                l2_leaf_reg=0.08,
                colsample_bylevel=0.75,
                random_strength=0.4,
                subsample=0.65,
                verbose=0,
            )
        ),
    },
    {
        "name": "d",
        "horizons": [slice(None)],
        "farms": [slice(None)],
        "hh4ofdays": list(range(0, 6)),
        "model": SqrtSquareCBRegressor(
            catboost.CatBoostRegressor(
                n_estimators=1200,
                bootstrap_type="Bernoulli",
                loss_function="MAPE",
                learning_rate=0.016,
                depth=8,
                l2_leaf_reg=0.08,
                colsample_bylevel=0.75,
                random_strength=0.4,
                subsample=0.65,
                verbose=0,
            )
        ),
    },
    {
        "name": "a.quant",
        "horizons": [slice(None)],
        "farms": [slice(None)],
        "hh4ofdays": [slice(None)],
        "model": QuantileCBRegressor(
            catboost.CatBoostRegressor(
                n_estimators=2000,
                bootstrap_type="Bernoulli",
                loss_function="MAPE",
                learning_rate=0.016,
                depth=8,
                l2_leaf_reg=0.08,
                colsample_bylevel=0.75,
                random_strength=0.4,
                subsample=0.65,
                verbose=0,
            )
        ),
    },
    {
        "name": "b.quant",
        "horizons": [slice(None)],
        "farms": list(range(1, 7)),
        "hh4ofdays": [slice(None)],
        "model": QuantileCBRegressor(
            catboost.CatBoostRegressor(
                n_estimators=1200,
                bootstrap_type="Bernoulli",
                loss_function="MAPE",
                learning_rate=0.016,
                depth=8,
                l2_leaf_reg=0.08,
                colsample_bylevel=0.75,
                random_strength=0.4,
                subsample=0.65,
                verbose=0,
            )
        ),
    },
    {
        "name": "c.quant",
        "horizons": list(range(0, 4)),
        "farms": [slice(None)],
        "hh4ofdays": [slice(None)],
        "model": QuantileCBRegressor(
            catboost.CatBoostRegressor(
                n_estimators=1200,
                bootstrap_type="Bernoulli",
                loss_function="MAPE",
                learning_rate=0.016,
                depth=8,
                l2_leaf_reg=0.08,
                colsample_bylevel=0.75,
                random_strength=0.4,
                subsample=0.65,
                verbose=0,
            )
        ),
    },
    {
        "name": "d.quant",
        "horizons": [slice(None)],
        "farms": [slice(None)],
        "hh4ofdays": list(range(0, 6)),
        "model": QuantileCBRegressor(
            catboost.CatBoostRegressor(
                n_estimators=1200,
                bootstrap_type="Bernoulli",
                loss_function="MAPE",
                learning_rate=0.016,
                depth=8,
                l2_leaf_reg=0.08,
                colsample_bylevel=0.75,
                random_strength=0.4,
                subsample=0.65,
                verbose=0,
            )
        ),
    },
    {
        "name": "a.power",
        "horizons": [slice(None)],
        "farms": [slice(None)],
        "hh4ofdays": [slice(None)],
        "model": PowerCBRegressor(
            catboost.CatBoostRegressor(
                n_estimators=2000,
                bootstrap_type="Bernoulli",
                loss_function="MAPE",
                learning_rate=0.016,
                depth=8,
                l2_leaf_reg=0.08,
                colsample_bylevel=0.75,
                random_strength=0.4,
                subsample=0.65,
                verbose=0,
            )
        ),
    },
    {
        "name": "b.power",
        "horizons": [slice(None)],
        "farms": list(range(1, 7)),
        "hh4ofdays": [slice(None)],
        "model": PowerCBRegressor(
            catboost.CatBoostRegressor(
                n_estimators=1200,
                bootstrap_type="Bernoulli",
                loss_function="MAPE",
                learning_rate=0.016,
                depth=8,
                l2_leaf_reg=0.08,
                colsample_bylevel=0.75,
                random_strength=0.4,
                subsample=0.65,
                verbose=0,
            )
        ),
    },
    {
        "name": "c.power",
        "horizons": list(range(0, 4)),
        "farms": [slice(None)],
        "hh4ofdays": [slice(None)],
        "model": PowerCBRegressor(
            catboost.CatBoostRegressor(
                n_estimators=1200,
                bootstrap_type="Bernoulli",
                loss_function="MAPE",
                learning_rate=0.016,
                depth=8,
                l2_leaf_reg=0.08,
                colsample_bylevel=0.75,
                random_strength=0.4,
                subsample=0.65,
                verbose=0,
            )
        ),
    },
    {
        "name": "d.power",
        "horizons": [slice(None)],
        "farms": [slice(None)],
        "hh4ofdays": list(range(0, 6)),
        "model": PowerCBRegressor(
            catboost.CatBoostRegressor(
                n_estimators=1200,
                bootstrap_type="Bernoulli",
                loss_function="MAPE",
                learning_rate=0.016,
                depth=8,
                l2_leaf_reg=0.08,
                colsample_bylevel=0.75,
                random_strength=0.4,
                subsample=0.65,
                verbose=0,
            )
        ),
    },
    {
        "name": "a.lgbm",
        "horizons": [slice(None)],
        "farms": [slice(None)],
        "hh4ofdays": [slice(None)],
        "model": SqrtSquareLGBMRegressor(
            lgb.LGBMRegressor(
                n_estimators=1800,
                learning_rate=0.02,
                objective="mape",
                colsample_bytree=0.8,
                num_leaves=50,
            )
        ),
    },
    {
        "name": "b.lgbm",
        "horizons": [slice(None)],
        "farms": list(range(1, 7)),
        "hh4ofdays": [slice(None)],
        "model": SqrtSquareLGBMRegressor(
            lgb.LGBMRegressor(
                n_estimators=1200,
                learning_rate=0.02,
                objective="mape",
                colsample_bytree=0.8,
                num_leaves=50,
            )
        ),
    },
    {
        "name": "c.lgbm",
        "horizons": list(range(0, 4)),
        "farms": [slice(None)],
        "hh4ofdays": [slice(None)],
        "model": SqrtSquareLGBMRegressor(
            lgb.LGBMRegressor(
                n_estimators=1200,
                learning_rate=0.02,
                objective="mape",
                colsample_bytree=0.8,
                num_leaves=50,
            )
        ),
    },
    {
        "name": "d.lgbm",
        "horizons": [slice(None)],
        "farms": [slice(None)],
        "hh4ofdays": list(range(0, 6)),
        "model": SqrtSquareLGBMRegressor(
            lgb.LGBMRegressor(
                n_estimators=1200,
                learning_rate=0.02,
                objective="mape",
                colsample_bytree=0.8,
                num_leaves=50,
            )
        ),
    },
]

features = feats + feats_com

for model in models:
    pred = ts_split_predict(data, split_start_dates, split_end_dates, features, model)
    pd.DataFrame(pred).reset_index(level=-1, drop=True).to_parquet(
        r"..\model_predictions\\" + model["name"] + ".parquet.gzip",
        engine="fastparquet",
        compression="gzip",
    )
