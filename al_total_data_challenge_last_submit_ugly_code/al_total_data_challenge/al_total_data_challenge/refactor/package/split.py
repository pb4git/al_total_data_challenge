import pandas as pd

def get_xval_split_boundaries(data, n_blocks):
    a = list(data.groupby('forecast_from')['ws'].count()[data.groupby('forecast_from')['ws'].count() == 72].index)
    a.append(a[-1]+pd.to_timedelta('84h'))
    test_dates_start = [b - pd.to_timedelta('11h') for b in a] #TODO should probably be renamed to say it's a forecast_from
    test_dates_end   = [b + pd.to_timedelta('36h') for b in a]

    # test periods : [test_dates_start, test_dates_end] both included

    # train window
    # data.set_index(['forecast_from', 'forecast_for']).sort_index().loc[(slice(test_dates_start[0]),slice(None)), 'ws']

    #train on [:xv_start[i] excluded [
    #predict on [xv_start[i] included : xv_end[i] included]
    xv_start = test_dates_start[::n_blocks]
    xv_end = [_ - pd.to_timedelta('1h') for _ in xv_start[1:]]
    xv_end.append(test_dates_end[-1])


    return xv_start, xv_end