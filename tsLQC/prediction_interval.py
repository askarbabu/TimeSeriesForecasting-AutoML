from tsLQC.constant import z_value
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd


def prediction_interval(ts, point_forecast):
    res = seasonal_decompose(ts['Value'], period=6, two_sided=True, extrapolate_trend=1)

    std = (ts['Value'] - res.trend).std()
    new_std = (point_forecast.reset_index().index.values ** 0.5) * std

    lower_forecast = pd.DataFrame(point_forecast['Value'] - z_value * new_std)
    upper_forecast = pd.DataFrame(point_forecast['Value'] + z_value * new_std)

    return lower_forecast, upper_forecast
