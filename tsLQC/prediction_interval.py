from tsLQC.constant import z_value, std_dev_estimation_len
import numpy as np
import pandas as pd


def prediction_interval(ts, point_forecast, model):
    t = len(ts)
    h = 1 + point_forecast.reset_index().index.values

    model.forecast_length = std_dev_estimation_len
    back_forecast = model.back_forecast(n_splits='auto').forecast.iloc[-std_dev_estimation_len:]
    actual_values = ts.iloc[-std_dev_estimation_len:]
    errors = back_forecast - actual_values
    std_dev_estimated = errors.values.std()

    new_std = std_dev_estimated * np.sqrt(h * (1 + (h/t)))

    lower_forecast = pd.DataFrame(point_forecast['Value'] - z_value * new_std)
    upper_forecast = pd.DataFrame(point_forecast['Value'] + z_value * new_std)

    return lower_forecast, upper_forecast
