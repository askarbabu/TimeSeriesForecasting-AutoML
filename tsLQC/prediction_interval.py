import numpy as np
from pandas import Series
from autots import AutoTS
from typing import Tuple
from statsmodels.tsa.seasonal import seasonal_decompose

STD_DEV_ESTIMATION_LEN = 4
Z_VALUE = 1.64


def prediction_interval(ts: Series, point_forecast: Series, model: AutoTS) -> Tuple[Series, Series]:
    t = len(ts)
    h = 1 + point_forecast.reset_index().index.values

    model.forecast_length = STD_DEV_ESTIMATION_LEN
    back_forecast = model.back_forecast(n_splits='auto').forecast.iloc[-STD_DEV_ESTIMATION_LEN:]
    actual_values = ts.interpolate(limit_direction='both').iloc[-STD_DEV_ESTIMATION_LEN:]
    errors = back_forecast - actual_values
    std_dev_estimated_from_fit = errors.values.std()

    std_dev_estimated_from_ts = seasonal_decompose(ts.interpolate(limit_direction='both'), period=6, two_sided=True,
                                                   extrapolate_trend=1).resid.std()

    std_dev_estimated = np.max([i for i in [std_dev_estimated_from_fit, std_dev_estimated_from_ts]
                                if not (np.isnan(i) or np.isinf(i))])
    new_std = std_dev_estimated * np.sqrt(h * (1 + (h/t)))

    lower_forecast = point_forecast - Z_VALUE * new_std
    upper_forecast = point_forecast + Z_VALUE * new_std

    return lower_forecast, upper_forecast
