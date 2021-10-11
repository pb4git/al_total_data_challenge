import pandas as pd


def get_timeseries_split_boundaries(data, n_blocks):

    # Returns a list of start dates, and a list of end dates that cut the dataset similarly to sklearn's TimeSeriesSplit,
    # but follows the 48h+36h structure of the public/private dataset.

    # Usage :
    # for i in range(n_blocks):
    #   - fit()     on the interval  [:xv_start[i] excluded [
    #   - predict() on the interval  [xv_start[i] included : xv_end[i] included]

    #######################

    # If a forecast_from has only 72 valid values, this represents 12 values per farm.
    # Given the train/test split chosen by organizers, this indicates that we are right at the end of a 48h test period
    forecasts_from_end_test_period = list(
        data.groupby("forecast_from")["ws"]
        .count()[data.groupby("forecast_from")["ws"].count() == 72]
        .index
    )
    forecasts_from_end_test_period.append(forecasts_from_end_test_period[-1] + pd.to_timedelta("84h"))
    
    test_dates_start = [
        date - pd.to_timedelta("11h") for date in forecasts_from_end_test_period
    ]  
    test_dates_end = [date + pd.to_timedelta("36h") for date in forecasts_from_end_test_period]

    xv_start = test_dates_start[::n_blocks]
    xv_end = [_ - pd.to_timedelta("1h") for _ in xv_start[1:]]
    xv_end.append(test_dates_end[-1])

    return xv_start, xv_end
