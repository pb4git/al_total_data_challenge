
import pandas as pd
from package.SqrtSquareCBRegressor import SqrtSquareCBRegressor, QuantileCBRegressor, PowerCBRegressor, SqrtSquareLGBMRegressor
from package.featureengineering import make_non_cyclic_categorical
from sklearn.metrics import mean_absolute_error
import catboost
import lightgbm as lgb

def ts_split_predict(data, split_dates_start, split_dates_end, features, model_param):
    #todo make a copy of all features that are now in the index

    data = data.copy().dropna(subset=features)

    data['farm_number_copy'] = data['farm_number']
    data['horizon_copy'] = data['horizon']

    print(model_param)

    for s, e in zip(split_dates_start, split_dates_end):
        data = make_non_cyclic_categorical(data, s) # create the variable, but must still be called somewhere else !!!                    

        data = data.set_index(['forecast_for', 'forecast_from', 'is_test', 'horizon_copy', 'farm_number_copy', 'hh4']).sort_index()


        for horizon in model_param['horizons']:
            print('horizon', horizon)
            for farm_number in model_param['farms']:
                print('farm', farm_number)
                for hh4ofday in model_param['hh4ofdays']:
                    print('hh4ofday', hh4ofday)

                    X_t = data.loc[(slice(s - pd.to_timedelta('1h')),
                                    slice(None),
                                    False,
                                    horizon,
                                    farm_number,
                                    hh4ofday),
                            features].copy()
                    
                    y_t = data.loc[(slice(s - pd.to_timedelta('1h')),
                                    slice(None),
                                    False,
                                    horizon,
                                    farm_number,
                                    hh4ofday),
                            'wp'].copy()

                    X_p = data.loc[(slice(s, e),
                                    slice(None),
                                    slice(None),
                                    horizon,
                                    farm_number,
                                    hh4ofday),
                            features]

                    #todo : use better parameters with optuna
                    cb = SqrtSquareCBRegressor(catboost.CatBoostRegressor
                                                    (
                                                    **model_param['catboost_param'],
                                                    verbose=0,
                                                    )
                                                )
                    
                    cb.fit(X_t, y_t)

                    data.loc[(  slice(s, e),
                                slice(None),
                                slice(None),
                                horizon,
                                farm_number,
                                hh4ofday),
                            model_param['name']] = cb.predict(X_p)
                    
        
        data = data.reset_index()
    

    data = data.set_index(['forecast_for', 'farm_number', 'horizon', 'forecast_from', 'is_test']).sort_index()
    
    mask = ~data[[model_param['name'], 'wp']].isna().any(axis=1)
    mask &= data.index.get_level_values(0).year >= 2011
    
    print(mean_absolute_error(data[mask][model_param['name']], data[mask]['wp']))
    
    print(mean_absolute_error(
        data[mask]['wp'].sort_index().groupby(level=[0, 1]).first(),
        data[mask][model_param['name']].sort_index().groupby(level=[0, 1]).first()
                   ))
    
    pred = data[model_param['name']].copy()
    data = data.reset_index().drop(model_param['name'], axis=1)
    return pred

def ts_split_predict_quant(data, split_dates_start, split_dates_end, features, model_param):
    #todo make a copy of all features that are now in the index

    data = data.copy().dropna(subset=features)

    data['farm_number_copy'] = data['farm_number']
    data['horizon_copy'] = data['horizon']

    print(model_param)

    for s, e in zip(split_dates_start, split_dates_end):
        data = make_non_cyclic_categorical(data, s) # create the variable, but must still be called somewhere else !!!                    

        data = data.set_index(['forecast_for', 'forecast_from', 'is_test', 'horizon_copy', 'farm_number_copy', 'hh4']).sort_index()


        for horizon in model_param['horizons']:
            print('horizon', horizon)
            for farm_number in model_param['farms']:
                print('farm', farm_number)
                for hh4ofday in model_param['hh4ofdays']:
                    print('hh4ofday', hh4ofday)

                    X_t = data.loc[(slice(s - pd.to_timedelta('1h')),
                                    slice(None),
                                    False,
                                    horizon,
                                    farm_number,
                                    hh4ofday),
                            features].copy()
                    
                    y_t = data.loc[(slice(s - pd.to_timedelta('1h')),
                                    slice(None),
                                    False,
                                    horizon,
                                    farm_number,
                                    hh4ofday),
                            'wp'].copy()

                    X_p = data.loc[(slice(s, e),
                                    slice(None),
                                    slice(None),
                                    horizon,
                                    farm_number,
                                    hh4ofday),
                            features]

                    #todo : use better parameters with optuna
                    cb = QuantileCBRegressor(catboost.CatBoostRegressor
                                                    (
                                                    **model_param['catboost_param'],
                                                    verbose=0,
                                                    )
                                                )
                    
                    cb.fit(X_t, y_t)

                    data.loc[(  slice(s, e),
                                slice(None),
                                slice(None),
                                horizon,
                                farm_number,
                                hh4ofday),
                            model_param['name']] = cb.predict(X_p)
                    
        
        data = data.reset_index()
    

    data = data.set_index(['forecast_for', 'farm_number', 'horizon', 'forecast_from', 'is_test']).sort_index()
    
    mask = ~data[[model_param['name'], 'wp']].isna().any(axis=1)
    mask &= data.index.get_level_values(0).year >= 2011
    
    print(mean_absolute_error(data[mask][model_param['name']], data[mask]['wp']))
    
    print(mean_absolute_error(
        data[mask]['wp'].sort_index().groupby(level=[0, 1]).first(),
        data[mask][model_param['name']].sort_index().groupby(level=[0, 1]).first()
                   ))
    
    pred = data[model_param['name']].copy()
    data = data.reset_index().drop(model_param['name'], axis=1)
    return pred

def ts_split_predict_power(data, split_dates_start, split_dates_end, features, model_param):
    #todo make a copy of all features that are now in the index

    data = data.copy().dropna(subset=features)

    data['farm_number_copy'] = data['farm_number']
    data['horizon_copy'] = data['horizon']

    print(model_param)

    for s, e in zip(split_dates_start, split_dates_end):
        data = make_non_cyclic_categorical(data, s) # create the variable, but must still be called somewhere else !!!                    

        data = data.set_index(['forecast_for', 'forecast_from', 'is_test', 'horizon_copy', 'farm_number_copy', 'hh4']).sort_index()


        for horizon in model_param['horizons']:
            print('horizon', horizon)
            for farm_number in model_param['farms']:
                print('farm', farm_number)
                for hh4ofday in model_param['hh4ofdays']:
                    print('hh4ofday', hh4ofday)

                    X_t = data.loc[(slice(s - pd.to_timedelta('1h')),
                                    slice(None),
                                    False,
                                    horizon,
                                    farm_number,
                                    hh4ofday),
                            features].copy()
                    
                    y_t = data.loc[(slice(s - pd.to_timedelta('1h')),
                                    slice(None),
                                    False,
                                    horizon,
                                    farm_number,
                                    hh4ofday),
                            'wp'].copy()

                    X_p = data.loc[(slice(s, e),
                                    slice(None),
                                    slice(None),
                                    horizon,
                                    farm_number,
                                    hh4ofday),
                            features]

                    #todo : use better parameters with optuna
                    cb = PowerCBRegressor(catboost.CatBoostRegressor
                                                    (
                                                    **model_param['catboost_param'],
                                                    verbose=0,
                                                    )
                                                )
                    
                    cb.fit(X_t, y_t)

                    data.loc[(  slice(s, e),
                                slice(None),
                                slice(None),
                                horizon,
                                farm_number,
                                hh4ofday),
                            model_param['name']] = cb.predict(X_p)
                    
        
        data = data.reset_index()
    

    data = data.set_index(['forecast_for', 'farm_number', 'horizon', 'forecast_from', 'is_test']).sort_index()
    
    mask = ~data[[model_param['name'], 'wp']].isna().any(axis=1)
    mask &= data.index.get_level_values(0).year >= 2011
    
    print(mean_absolute_error(data[mask][model_param['name']], data[mask]['wp']))
    
    print(mean_absolute_error(
        data[mask]['wp'].sort_index().groupby(level=[0, 1]).first(),
        data[mask][model_param['name']].sort_index().groupby(level=[0, 1]).first()
                   ))
    
    pred = data[model_param['name']].copy()
    data = data.reset_index().drop(model_param['name'], axis=1)
    return pred


def ts_split_predict_2(data, split_dates_start, split_dates_end, features_per_farm, features_common, model_param):

    data = data.copy().dropna(subset=features_per_farm+features_common)

    data['horizon_copy'] = data['horizon']

    print(model_param)

    # just to prepare the empty dataframe d
    from itertools import repeat
    a = data.pivot(index=['forecast_for', 'forecast_from', 'is_test', 'horizon_copy', 'hh4'], columns='farm_number')
    d = pd.DataFrame(index=a.index)

    for farm_number in range(1, 7):
        print('farm', farm_number)
        features = [(feature, fn) for fn in model_param['farms_to_use_as_features'][farm_number] for feature in features_per_farm]

        for s, e in zip(split_dates_start, split_dates_end):
            data = make_non_cyclic_categorical(data, s) # create the variable, but must still be called somewhere else !!!                    
            
            from itertools import repeat
            a = data.pivot(index=['forecast_for', 'forecast_from', 'is_test', 'horizon_copy', 'hh4'], columns='farm_number')
            b = pd.concat([a[features], a[list(zip(features_common, repeat(1)))].droplevel(1, axis=1)], axis=1)
            c = a['wp']
            
            for horizon in model_param['horizons']:
                print('horizon', horizon)
                for hh4ofday in model_param['hh4ofdays']:
                    print('hh4ofday', hh4ofday)

                    X_t = b.loc[(slice(s - pd.to_timedelta('1h')),
                                    slice(None),
                                    False,
                                    horizon,
                                    hh4ofday),
                            :].copy()
                    
                    y_t = c.loc[(slice(s - pd.to_timedelta('1h')),
                                    slice(None),
                                    False,
                                    horizon,
                                    hh4ofday),
                            farm_number].copy()

                    X_p = b.loc[(slice(s, e),
                                    slice(None),
                                    slice(None),
                                    horizon,
                                    hh4ofday),
                            :].copy()

                    #todo : use better parameters with optuna
                    cb = SqrtSquareCBRegressor(catboost.CatBoostRegressor
                                                    (
                                                    **model_param['catboost_param'],
                                                    verbose=0,
                                                    )
                                                )
                    
                    cb.fit(X_t, y_t)

                    d.loc[(slice(s, e),
                                    slice(None),
                                    slice(None),
                                    horizon,
                                    hh4ofday),
                          'pred'+str(farm_number)] = cb.predict(X_p)                  
            

    pred2 = d[['pred'+str(i) for i in range(1, 7)]]
    pred2.columns = pd.MultiIndex.from_product([[model_param['name']], list(range(1,7))], names=(None, 'farm_number'))
    pred2 = pred2.stack().sort_index()
    
    data = data.set_index(['forecast_for', 'forecast_from', 'is_test', 'horizon_copy', 'hh4', 'farm_number']).sort_index()
    data[model_param['name']] = pred2
    
    mask = ~data[[model_param['name'], 'wp']].isna().any(axis=1)
    mask &= data.index.get_level_values(0).year >= 2011
    
    print(mean_absolute_error(data[mask][model_param['name']], data[mask]['wp']))
    
    print(mean_absolute_error(
        data[mask]['wp'].sort_index().groupby(level=[0, 1]).first(),
        data[mask][model_param['name']].sort_index().groupby(level=[0, 1]).first()
                   ))
    
    pred = data[model_param['name']].copy()
    data = data.reset_index().drop(model_param['name'], axis=1)
    
    return pred.reset_index().rename(columns={'horizon_copy':'horizon'}).set_index(['forecast_for', 'farm_number', 'horizon', 'forecast_from', 'is_test']).sort_index()[model_param['name']]



def ts_split_predict_lgbm(data, split_dates_start, split_dates_end, features, model_param):
    #todo make a copy of all features that are now in the index

    data = data.copy().dropna(subset=features)

    data['farm_number_copy'] = data['farm_number']
    data['horizon_copy'] = data['horizon']

    print(model_param)

    for s, e in zip(split_dates_start, split_dates_end):
        data = make_non_cyclic_categorical(data, s) # create the variable, but must still be called somewhere else !!!                    

        data = data.set_index(['forecast_for', 'forecast_from', 'is_test', 'horizon_copy', 'farm_number_copy', 'hh4']).sort_index()


        for horizon in model_param['horizons']:
            print('horizon', horizon)
            for farm_number in model_param['farms']:
                print('farm', farm_number)
                for hh4ofday in model_param['hh4ofdays']:
                    print('hh4ofday', hh4ofday)

                    X_t = data.loc[(slice(s - pd.to_timedelta('1h')),
                                    slice(None),
                                    False,
                                    horizon,
                                    farm_number,
                                    hh4ofday),
                            features].copy()
                    
                    y_t = data.loc[(slice(s - pd.to_timedelta('1h')),
                                    slice(None),
                                    False,
                                    horizon,
                                    farm_number,
                                    hh4ofday),
                            'wp'].copy()

                    X_p = data.loc[(slice(s, e),
                                    slice(None),
                                    slice(None),
                                    horizon,
                                    farm_number,
                                    hh4ofday),
                            features]

                    #todo : use better parameters with optuna
                    cb = SqrtSquareLGBMRegressor(lgb.LGBMRegressor
                                                    (
                                                    **model_param['lgbm_param']
                                                    )
                                                )
                    
                    cb.fit(X_t, y_t)

                    data.loc[(  slice(s, e),
                                slice(None),
                                slice(None),
                                horizon,
                                farm_number,
                                hh4ofday),
                            model_param['name']] = cb.predict(X_p)
                    
        
        data = data.reset_index()
    

    data = data.set_index(['forecast_for', 'farm_number', 'horizon', 'forecast_from', 'is_test']).sort_index()
    
    mask = ~data[[model_param['name'], 'wp']].isna().any(axis=1)
    mask &= data.index.get_level_values(0).year >= 2011
    
    print(mean_absolute_error(data[mask][model_param['name']], data[mask]['wp']))
    
    print(mean_absolute_error(
        data[mask]['wp'].sort_index().groupby(level=[0, 1]).first(),
        data[mask][model_param['name']].sort_index().groupby(level=[0, 1]).first()
                   ))
    
    pred = data[model_param['name']].copy()
    data = data.reset_index().drop(model_param['name'], axis=1)
    return pred

def ts_split_predict_2_lgbm(data, split_dates_start, split_dates_end, features_per_farm, features_common, model_param):

    data = data.copy().dropna(subset=features_per_farm+features_common)

    data['horizon_copy'] = data['horizon']

    print(model_param)

    # just to prepare the empty dataframe d
    from itertools import repeat
    a = data.pivot(index=['forecast_for', 'forecast_from', 'is_test', 'horizon_copy', 'hh4'], columns='farm_number')
    d = pd.DataFrame(index=a.index)

    for farm_number in range(1, 7):
        print('farm', farm_number)
        features = [(feature, fn) for fn in model_param['farms_to_use_as_features'][farm_number] for feature in features_per_farm]

        for s, e in zip(split_dates_start, split_dates_end):
            data = make_non_cyclic_categorical(data, s) # create the variable, but must still be called somewhere else !!!                    
            
            from itertools import repeat
            a = data.pivot(index=['forecast_for', 'forecast_from', 'is_test', 'horizon_copy', 'hh4'], columns='farm_number')
            b = pd.concat([a[features], a[list(zip(features_common, repeat(1)))].droplevel(1, axis=1)], axis=1)
            c = a['wp']
            
            for horizon in model_param['horizons']:
                print('horizon', horizon)
                for hh4ofday in model_param['hh4ofdays']:
                    print('hh4ofday', hh4ofday)

                    X_t = b.loc[(slice(s - pd.to_timedelta('1h')),
                                    slice(None),
                                    False,
                                    horizon,
                                    hh4ofday),
                            :].copy()

                    X_t.columns = list(range(len(X_t.columns)))
                    
                    y_t = c.loc[(slice(s - pd.to_timedelta('1h')),
                                    slice(None),
                                    False,
                                    horizon,
                                    hh4ofday),
                            farm_number].copy()

                    X_p = b.loc[(slice(s, e),
                                    slice(None),
                                    slice(None),
                                    horizon,
                                    hh4ofday),
                            :].copy()
                    X_p.columns = list(range(len(X_p.columns)))

                    #todo : use better parameters with optuna
                    cb = SqrtSquareLGBMRegressor(lgb.LGBMRegressor
                                                    (
                                                    **model_param['lgbm_param']
                                                    )
                                                )
                    
                    cb.fit(X_t, y_t)

                    d.loc[(slice(s, e),
                                    slice(None),
                                    slice(None),
                                    horizon,
                                    hh4ofday),
                          'pred'+str(farm_number)] = cb.predict(X_p)                  
            

    pred2 = d[['pred'+str(i) for i in range(1, 7)]]
    pred2.columns = pd.MultiIndex.from_product([[model_param['name']], list(range(1,7))], names=(None, 'farm_number'))
    pred2 = pred2.stack().sort_index()
    
    data = data.set_index(['forecast_for', 'forecast_from', 'is_test', 'horizon_copy', 'hh4', 'farm_number']).sort_index()
    data[model_param['name']] = pred2
    
    mask = ~data[[model_param['name'], 'wp']].isna().any(axis=1)
    mask &= data.index.get_level_values(0).year >= 2011
    
    print(mean_absolute_error(data[mask][model_param['name']], data[mask]['wp']))
    
    print(mean_absolute_error(
        data[mask]['wp'].sort_index().groupby(level=[0, 1]).first(),
        data[mask][model_param['name']].sort_index().groupby(level=[0, 1]).first()
                   ))
    
    pred = data[model_param['name']].copy()
    data = data.reset_index().drop(model_param['name'], axis=1)
    
    return pred.reset_index().rename(columns={'horizon_copy':'horizon'}).set_index(['forecast_for', 'farm_number', 'horizon', 'forecast_from', 'is_test']).sort_index()[model_param['name']]
