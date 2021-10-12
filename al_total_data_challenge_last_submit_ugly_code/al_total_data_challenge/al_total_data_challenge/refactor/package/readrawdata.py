import pandas as pd

def readrawdata(data_path = '/home/paul/Documents/ALTotal Challenge/data/phase2/'):
    forecast_datas = []

    for i in range(1, 7):
        forecast_data = pd.read_csv(f'{data_path}wp{i}.csv')
        forecast_data.date = pd.to_datetime(forecast_data.date, format='%Y%m%d%H')
        forecast_data['farm_number'] = i
        forecast_datas.append(forecast_data)

    data = pd.concat(forecast_datas[0:6]).dropna()
    data.rename(columns={'date':'forecast_from'}, inplace=True)
    data['forecast_from_hour'] = pd.Categorical(data['forecast_from'].dt.hour)
    data['forecast_for']       = data['forecast_from']+pd.to_timedelta('1h')*data['hors']
    data['horizon']            = (data['hors'] - 1) // 12      #0: [1-12h]  1: [13-24h]  2: [25-36h]  3: [37-48h]
    data['hour_in_b12']        = (data['hors'] - 1) % 12
    data['hour_in_b48']        = data['hors'] - 1
    data['farm_number']        = pd.Categorical(data['farm_number'])

    wp = pd.read_csv(f'{data_path}train.csv')
    wp['date'] = pd.to_datetime(wp.date, format='%Y%m%d%H')
    wp.set_index('date', inplace=True)
    wp.columns = pd.MultiIndex.from_product([['wp'], list(range(1,7))], names=(None, 'farm_number'))

    test_dates = pd.read_csv(f'{data_path}test.csv')
    test_dates.date = pd.to_datetime(test_dates.date, format='%Y%m%d%H')
    test_dates.set_index('date', inplace=True)

    data = data.set_index(['forecast_for', 'farm_number']).sort_index().join(other=wp.stack().reset_index().rename(columns={'date':'forecast_for'}).set_index(['forecast_for', 'farm_number']).sort_index()).set_index('horizon', append=True).sort_index().reset_index()                               

    return data, test_dates