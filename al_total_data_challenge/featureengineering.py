import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from al_total_data_challenge.feature_engineering_on_pivoted_data import make_sliding, make_derivatives, make_OLS_derivatives

def featureengineering(data):

    window_sizes = [3, 5, 7, 9, 11, 13, 15]

    data = make_cyclic_categorical(data)
    data = make_non_cyclic_categorical(data, pd.Timestamp('2009-07-01 01:00:00')) #This call creates the variable, but it must be made regularly at each split
    data = make_mixed_ws(data)
    data = make_neighbour_features(data)
    data = make_clustering_features(data)
    data = make_lagged_by_horizon(data)

    ##################################
    # Create most slid features
    for forecast_from_hour in data['forecast_from_hour'].unique(): #0h, 12h
        for window_size in window_sizes:
            for feature_to_slide in ['ws', 'u', 'v']:
                pivoted_data = data[data['forecast_from_hour'] == forecast_from_hour][[feature_to_slide, 'farm_number', 'forecast_from', 'hour_in_b48']].pivot(index=['hour_in_b48'], columns=['farm_number', 'forecast_from'])[feature_to_slide]

                data = data.set_index(['hour_in_b48', 'farm_number', 'forecast_from']).sort_index()
                slid_features = make_sliding(pivoted_data, feature_to_slide, window_size)
                data = pd.concat([data]+ [slid_feature.stack(level=[1,2]).sort_index() for slid_feature in slid_features], axis=1)
                data = data.reset_index()

    # hack : Previous lines create several columns, one for each "forecast_from_hour" --> merge those into only one column, remove all duplicates
    for col_name in data.columns[data.columns.duplicated()]:
        data[col_name] = data[col_name].mean(axis=1)
    data = data.loc[:,~data.columns.duplicated()]

    ##################################
    # Feature to evaluate typical weather prediction uncertainty around this date
    for forecast_from_hour in data['forecast_from_hour'].unique(): #0h, 12h
        for window_size in [3, 5]:
            for feature_to_slide in ['ws_delta_horizon']:
                pivoted_data = data[data['forecast_from_hour'] == forecast_from_hour][[feature_to_slide, 'farm_number', 'forecast_from', 'hour_in_b48']].pivot(index=['hour_in_b48'], columns=['farm_number', 'forecast_from'])[feature_to_slide]

                data = data.set_index(['hour_in_b48', 'farm_number', 'forecast_from']).sort_index()
                slid_features = make_sliding(pivoted_data, feature_to_slide, window_size)
                data = pd.concat([data]+ [slid_feature.stack(level=[1,2]).sort_index() for slid_feature in slid_features], axis=1)
                data = data.reset_index()

    # hack : Previous lines create several columns, one for each "forecast_from_hour" --> merge those into only one column, remove all duplicates
    for col_name in data.columns[data.columns.duplicated()]:
        data[col_name] = data[col_name].mean(axis=1)
    data = data.loc[:,~data.columns.duplicated()]

    ##################################

    for features_to_derive in ['ws', 'u', 'v']:
        pivoted_data = data[[features_to_derive, 'farm_number', 'forecast_from', 'hour_in_b48']].pivot(index=['hour_in_b48'], columns=['farm_number', 'forecast_from'])[features_to_derive]
        data = data.set_index(['hour_in_b48', 'farm_number', 'forecast_from']).sort_index()
        data = pd.concat([data]+ [derivatives.stack(level=[1,2]).sort_index() for derivatives in make_derivatives(pivoted_data, features_to_derive)], axis=1)
        data = pd.concat([data]+ [derivatives.stack(level=[1,2]).sort_index() for derivatives in make_OLS_derivatives(pivoted_data, features_to_derive)], axis=1)
        data = data.reset_index()

    for i in window_sizes:
        data['uv_mean_w'+str(i)] = np.sqrt(data['u_mean_w'+str(i)]**2 + data['v_mean_w'+str(i)]**2)
        data['uv_std_w'+str(i)]  = np.sqrt(data['u_std_w'+str(i)]**2 + data['v_std_w'+str(i)]**2)
        data['wd_mean_w'+str(i)] = np.arctan2(data['u_mean_w'+str(i)], data['v_mean_w'+str(i)])

    ##################################

    data['duv_dot___uvmean'] =          data['u_d1_cl']*data['u_mean_w3'] + data['v_d1_cl']*data['v_mean_w3']
    data['duv_cross_uvmean'] =          data['u_d1_cl']*data['v_mean_w3'] - data['v_d1_cl']*data['u_mean_w3']
    data['duv_dot___uvmean_n'] =        data['duv_dot___uvmean'] / data['uv_mean_w3']
    data['duv_dot___uvmean_n_abs'] =    (data['duv_dot___uvmean'] / data['uv_mean_w3']).abs()
    data['duv_cross_uvmean_n'] =        data['duv_cross_uvmean'] / data['uv_mean_w3']

    data['d2uv_dot___uvmean'] =         data['u_d2_cl']*data['u_mean_w3'] + data['v_d2_cl']*data['v_mean_w3']
    data['d2uv_cross_uvmean'] =         data['u_d2_cl']*data['v_mean_w3'] - data['v_d2_cl']*data['u_mean_w3']
    data['d2uv_dot___uvmean_n'] =       data['d2uv_dot___uvmean'] / data['uv_mean_w3']
    data['d2uv_dot___uvmean_n_abs'] =   (data['d2uv_dot___uvmean'] / data['uv_mean_w3']).abs()
    data['d2uv_cross_uvmean_n'] =       data['d2uv_cross_uvmean'] / data['uv_mean_w3']

    data['duv_norm'] =                  np.sqrt(data['u_d1_cl']**2 + data['v_d1_cl']**2)
    data['d2uv_norm'] =                 np.sqrt(data['u_d2_cl']**2 + data['v_d2_cl']**2)
    data['d3uv_norm'] =                 np.sqrt(data['u_d3_cl']**2 + data['v_d3_cl']**2)

    ##################################

    for window_size in window_sizes:
        cols = ['ws_mean_w'+str(window_size),
                'ws_std_w'+str(window_size),
                'uv_mean_w'+str(window_size),
                'uv_std_w'+str(window_size)]
        lr = Ridge()
        d = data[(data['forecast_for'].dt.year<2011) & (data['wp'] < 0.8) & (data['wp'] > 0.05)][cols+['wp']].dropna()
        lr.fit(d[cols], d['wp'])
        data.loc[~data[cols].isna().any(axis=1), 'wsuv_w'+str(window_size)] = lr.predict(data[~data[cols].isna().any(axis=1)][cols])
    
    ##################################


    return data

def make_clustering_features(data):

    # Clusters : 
    # - per farm, 12x 1h forecast on windspeed
    # - per farm, 12x 1h forecast on u, v
    # - per 1h forecast, 6x farm forecast on windspeed
    # - per 1h forecast, 6x farm forecast on u, v

    for n_clusters in [16, 32, 48]:
        data2 = data.pivot(index=['forecast_from', 'horizon', 'farm_number'], columns='hour_in_b12').sort_index()
        km_blocks = pd.DataFrame(index=data2.index, data={'km_ws_block_'+str(n_clusters):KMeans(n_clusters=n_clusters).fit_predict(data2['ws']*np.array([1.65, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.65])),
                                                        'km_uv_block_'+str(n_clusters):KMeans(n_clusters=n_clusters).fit_predict(data2[['u', 'v']]*np.array([1.3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.3, 1.3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.3]))})
        data2 = data2.stack().reset_index().set_index(['forecast_from', 'horizon', 'farm_number']).sort_index()
        data2 = data2.join(other=km_blocks)
        data = data.set_index(['forecast_from', 'horizon', 'farm_number']).sort_index()
        data['km_ws_block_'+str(n_clusters)] = pd.Categorical(data2['km_ws_block_'+str(n_clusters)])
        data['km_uv_block_'+str(n_clusters)] = pd.Categorical(data2['km_uv_block_'+str(n_clusters)])
        data.reset_index(inplace=True)
    
        data2 = data.pivot(index=['forecast_from', 'horizon', 'hour_in_b12'], columns='farm_number').sort_index()
        km_farms = pd.DataFrame(index=data2.index, data={'km_ws_farms_'+str(n_clusters):KMeans(n_clusters=n_clusters).fit_predict(data2['ws']),
                                                        'km_uv_farms_'+str(n_clusters):KMeans(n_clusters=n_clusters).fit_predict(data2[['u', 'v']])})
        data2 = data2.stack().reset_index().set_index(['forecast_from', 'horizon', 'hour_in_b12']).sort_index()
        data2 = data2.join(other=km_farms)
        data = data.set_index(['forecast_from', 'horizon', 'farm_number']).sort_index()
        data['km_ws_farms_'+str(n_clusters)] = pd.Categorical(data2['km_ws_farms_'+str(n_clusters)])
        data['km_uv_farms_'+str(n_clusters)] = pd.Categorical(data2['km_uv_farms_'+str(n_clusters)])
        data.reset_index(inplace=True)
    return data


def make_neighbour_features(data):
    # Try to interpolate u, v, ws for a farm, knowing only the u, v, ws value for the neighbouring farms
    data = data.set_index(['forecast_for', 'horizon', 'farm_number']).sort_index()

    for fn in range(1, 7):
        farms = list(range(1, 7))
        farms.remove(fn)

        xt = data.loc[(slice(pd.to_datetime('2010-12-31 12:00:00')), slice(None), farms),:].unstack(level=2)[['u', 'v', 'ws']]
        yt = data.loc[(slice(pd.to_datetime('2010-12-31 12:00:00')), slice(None), [fn]),:][['u', 'v', 'ws']]
        x = data.loc[(slice(None), slice(None), farms),:].unstack(level=2)[['u', 'v', 'ws']]
        lr = Ridge(alpha=100)
        lr.fit(xt, yt)
        data.loc[(slice(None), slice(None), [fn]), ['u_nei_lr', 'v_nei_lr', 'ws_nei_lr']] = lr.predict(x)

    data['ws_nei_lr_calc'] = np.sqrt(data['u_nei_lr']**2+data['v_nei_lr']**2)

    data = data.reset_index().set_index(['forecast_for', 'farm_number', 'horizon']).sort_index()                              

    ws_mean = data.unstack(level=1)['ws'].mean(axis=1)
    u_mean = data.unstack(level=1)['u'].mean(axis=1)
    v_mean = data.unstack(level=1)['v'].mean(axis=1)
    
    data = data.reset_index().set_index(['forecast_for', 'horizon'])
    data['ws_nei_mean'] = ws_mean
    data['u_nei_mean'] = u_mean
    data['v_nei_mean'] = v_mean
    data = data.reset_index().set_index(['forecast_for', 'farm_number']).sort_index()
    
    lr = Ridge(alpha=100)
    lr.fit(data.loc[(slice(pd.to_datetime('2010-12-31 23:00:00')), slice(None)), :].dropna()[['ws', 'ws_nei_lr', 'ws_nei_lr_calc', 'ws_nei_mean']], data.loc[(slice(pd.to_datetime('2010-12-31 23:00:00')), slice(None)), :].dropna()['wp'])
    data['ws_lr'] = np.sum(data[['ws', 'ws_nei_lr', 'ws_nei_lr_calc', 'ws_nei_mean']].values * lr.coef_ / np.sum(lr.coef_),axis=1)
    
    lr = Ridge(alpha=100)
    lr.fit(data.loc[(slice(pd.to_datetime('2010-12-31 23:00:00')), slice(None)), :].dropna()[['ws', 'ws_nei_mean']], data.loc[(slice(pd.to_datetime('2010-12-31 23:00:00')), slice(None)), :].dropna()['wp'])
    data['ws_lr2'] = np.sum(data[['ws', 'ws_nei_mean']].values * lr.coef_ / np.sum(lr.coef_),axis=1)
    
    data = data.reset_index()
    return data

def make_lagged_by_horizon(data):

    # Create features containing the 12h older prediction for the same forecast_for and farm_number
    # Answers the question "Have the predictions been stable for this or variable ?"

    a = data.pivot(index='horizon', columns=['forecast_for', 'farm_number'], values=['ws', 'u', 'v']).sort_index()
    for feat_name in ['ws', 'u', 'v']:
        b = a[feat_name].shift(-1)
        b = b.fillna(value=a[feat_name])
        c = b.stack()
        d = c.stack().sort_index()
        data = data.set_index(['horizon', 'farm_number', 'forecast_for']).sort_index()
        data[feat_name+'_horizon'] = d
        data[feat_name+'_delta_horizon'] = data[feat_name] - data[feat_name+'_horizon']
        data = data.reset_index()
    data['uv_norm_delta_horizon'] = np.sqrt(data['u_delta_horizon']**2 + data['v_delta_horizon']**2)
    data['uv_delta_horizon_dot_uvn'] = (data['u_delta_horizon'] * data['u'] + data['v_delta_horizon'] * data['v'])/(0.1+data['ws'])
    data['uv_delta_horizon_cross_uvn'] = (data['u_delta_horizon'] * data['v'] - data['v_delta_horizon'] * data['u'])/(0.1+data['ws'])
    data['uv_delta_horizon_cross_uvn_abs'] = ((data['u_delta_horizon'] * data['v'] - data['v_delta_horizon'] * data['u']).abs())/(0.1+data['ws'])  
    return data

def make_mixed_ws(data):

    # Try to create a "mixed" windspeed for each farm, based on windspeed and winddirection predictions for all farms.

    data.set_index(['forecast_for', 'horizon', 'farm_number'], inplace=True)
    ws = data['ws'].unstack()
    u = data['u'].unstack()
    v = data['v'].unstack()
    wp = data['wp'].unstack()
    ws_copy = ws.copy()
    
    for fn in range(1, 7):
        dot = (u.multiply(u[fn], axis=0) + v.multiply(v[fn], axis=0)) / ws
        cross_abs = (u.multiply(v[fn], axis=0) - v.multiply(u[fn], axis=0)).abs() / ws
        
        ws_dot_cross = pd.concat([ws, dot, cross_abs], axis=1)
        
        X = ws_dot_cross.loc[(slice(pd.to_datetime('2010-12-31 23:00:00')), slice(None), slice(None)),:]
        y = wp.loc[(slice(pd.to_datetime('2010-12-31 23:00:00')), slice(None), slice(None)),fn]
        
        lr = Ridge(alpha=100)
        lr.fit(X[(y > 0.05) & (y < 0.85)], y[(y > 0.05) & (y < 0.85)])
        
        ws_copy[fn] = lr.predict(ws_dot_cross)

    data['ws_mixed'] = ws_copy.stack()
    
    # Totally arbitrary re-normalization to the same mean and deviation as a normal windspeed
    data['ws_mixed'] = (data['ws_mixed'] - data.loc[(slice(pd.to_datetime('2010-12-31 23:00:00')), slice(None), slice(None)), 'ws_mixed'].mean()) / data.loc[(slice(pd.to_datetime('2010-12-31 23:00:00')), slice(None), slice(None)), 'ws_mixed'].std()
    data['ws_mixed'] = data['ws_mixed'] * data.loc[(slice(pd.to_datetime('2010-12-31 23:00:00')), slice(None), slice(None)), 'ws'].std() + data.loc[(slice(pd.to_datetime('2010-12-31 23:00:00')), slice(None), slice(None)), 'ws'].mean()
    data['ws_mixed'] = (data['ws_mixed'] + data.loc[(slice(pd.to_datetime('2010-12-31 23:00:00')), slice(None), slice(None)), 'ws_mixed'].min()).clip(lower=0)
    
    data = data.reset_index()    
    return data

def make_cyclic_categorical(data):

    # Obviously not all will be used...
    
    data['mm'] =               pd.Categorical(     data['forecast_for'].dt.month)
    data['mm2'] =              pd.Categorical(     ((data['forecast_for'].dt.month-1)//2))
    data['mm3'] =              pd.Categorical(     ((data['forecast_for'].dt.month-1)//3))
    data['mm4'] =              pd.Categorical(     ((data['forecast_for'].dt.month-1)//4))
    data['hh'] =               pd.Categorical(     data['forecast_for'].dt.hour)
    data['hh2'] =              pd.Categorical(     ((data['forecast_for'].dt.hour + 23)%24) // 2)
    data['hh3'] =              pd.Categorical(     ((data['forecast_for'].dt.hour + 23)%24) // 3)
    data['hh4'] =              pd.Categorical(     ((data['forecast_for'].dt.hour + 23)%24) // 4)
    data['hh6'] =              pd.Categorical(     ((data['forecast_for'].dt.hour + 23)%24) // 6)
    data['am_pm'] =            pd.Categorical(     ((data['forecast_for'].dt.hour + 23) % 24 ) // 12) #am pm



    data['mmhh'] =             pd.Categorical(     data['mm'].astype(int)*24 +         data['hh'].astype(int))
    data['mmhh2'] =            pd.Categorical(     data['mm'].astype(int)*12 +         data['hh2'].astype(int))
    data['mm2hh2'] =           pd.Categorical(     data['mm2'].astype(int)*12 +        data['hh2'].astype(int))
    data['mm3hh4'] =           pd.Categorical(     data['mm3'].astype(int)*6 +         data['hh4'].astype(int))
    data['mmhh3'] =            pd.Categorical(     data['mm'].astype(int)*8 +          data['hh3'].astype(int))
    data['mm2hh3'] =           pd.Categorical(     data['mm2'].astype(int)*8 +         data['hh3'].astype(int))

    first_date = pd.Timestamp('2009-07-01 01:00:00')
    data['f_for_12h_group'] = pd.Categorical(((data['forecast_for'] - first_date) / pd.Timedelta('12h')).astype(int))

    return data


def make_non_cyclic_categorical(data, start):
    
    # "special" features : used the same way as "yy" but takes into account the absence of fit() for future data.
    # If the fit is done on data from 2010 and 2011, a "basic" yy feature would be ininterpretable for the model when is takes the value 2012 or 2013.
    # ==> Future values are replaced by the latest value present in the training dataset.

    # Can be done at any frequency : yy is early, yymm4 is three times per year

    # This transformation does not change anything if a decision tree uses the feature as numerical.
    # However, if the feature is declared categorical for catboost, the transformation is useful.
    data['yy'] = ((start-data['forecast_for'])//pd.to_timedelta('8760h')).clip(lower=0)
    data['yymm4'] = ((start-data['forecast_for'])//pd.to_timedelta('2920h')).clip(lower=0)
    return data
