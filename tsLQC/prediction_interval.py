from tsLQC.constant import z_value, std_dev_estimation_len
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


def prediction_interval(ts, point_forecast, model):
    t = len(ts)
    h = 1 + point_forecast.reset_index().index.values

    model.forecast_length = std_dev_estimation_len
    back_forecast = model.back_forecast(n_splits='auto').forecast.iloc[-std_dev_estimation_len:]
    actual_values = ts.interpolate().iloc[-std_dev_estimation_len:]
    errors = back_forecast - actual_values
    std_dev_estimated_from_fit = errors.values.std()

    std_dev_estimated_from_ts = seasonal_decompose(ts.interpolate()['Value'], period=6, two_sided=True,
                                                   extrapolate_trend=1).resid.std()

    std_dev_estimated = np.max([i for i in [std_dev_estimated_from_fit, std_dev_estimated_from_ts]
                                if not (np.isnan(i) or np.isinf(i))])
    new_std = std_dev_estimated * np.sqrt(h * (1 + (h/t)))

    lower_forecast = point_forecast['Value'] - z_value * new_std
    upper_forecast = point_forecast['Value'] + z_value * new_std

    return lower_forecast, upper_forecast
