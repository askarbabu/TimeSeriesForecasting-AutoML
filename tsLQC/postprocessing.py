from pandas import Series
import numpy as np
from autots import AutoTS
from typing import Tuple
from tsLQC.constant import DATE_COL, VALUE_COL
from statsmodels.tsa.seasonal import seasonal_decompose


def bad_forecast_handling(ts: Series, point_forecast: Series, model: AutoTS) -> Tuple[Series, AutoTS]:
    ts_trend = seasonal_decompose(ts.interpolate(limit_direction='both'), period=6, two_sided=True,
                                  extrapolate_trend=1).trend
    ts_trend_slope = (ts_trend[-1] - ts_trend[-12]) / 12

    forecasted_trend = seasonal_decompose(point_forecast, period=6, two_sided=True, extrapolate_trend=1).trend
    forecasted_trend_slope = (forecasted_trend.iloc[11] - forecasted_trend.iloc[0]) / 12

    slope_criteria = ((forecasted_trend_slope - ts_trend_slope)/ts_trend_slope < -0.75) and (ts_trend_slope > 0)
    distinct_value_criteria = len(set(np.round(forecasted_trend.values))) < 36

    if slope_criteria or distinct_value_criteria:
        model = AutoTS(forecast_length=4,
                       no_negatives=True,
                       ensemble=None,
                       max_generations=3,
                       model_list=['ETS'],
                       verbose=1,
                       num_validations=3)
        model = model.fit(ts.reset_index(), date_col=DATE_COL, value_col=VALUE_COL, id_col=None)
        point_forecast = model.predict(48).forecast[VALUE_COL]

    return point_forecast, model


def noise_addition(ts: Series, point_forecast: Series) -> Series:
    res = seasonal_decompose(ts.interpolate(limit_direction='both'), period=6, two_sided=True,
                             extrapolate_trend=1).resid
    noise = np.random.normal(res.mean(), res.std(), len(point_forecast))
    output_values = point_forecast + noise * np.linspace(0, 1, len(point_forecast))

    return output_values
