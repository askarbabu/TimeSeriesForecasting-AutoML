from pandas import Series
import numpy as np
from autots import AutoTS
from typing import Tuple
from tsLQC.constant import DATE_COL, VALUE_COL, FORECAST_PERIOD, VERBOSE, NO_NEGATIVES
from statsmodels.tsa.seasonal import seasonal_decompose


PERIOD_FOR_SLOPE = 12
FORECASTED_TREND_REDUCTION_THR = -0.75
TREND_DISTINCT_VALUE_THR = 36
VALIDATION_POINTS = 4
MODEL_LIST = ['ETS']
MAX_GENERATIONS = 3
NUM_VALIDATIONS = 3


def bad_forecast_handling(ts: Series, point_forecast: Series, model: AutoTS) -> Tuple[Series, AutoTS]:
    ts_trend = seasonal_decompose(ts.interpolate(limit_direction='both'), period=6, two_sided=True,
                                  extrapolate_trend=1).trend
    ts_trend_slope = (ts_trend[-1] - ts_trend[-PERIOD_FOR_SLOPE]) / PERIOD_FOR_SLOPE

    forecasted_trend = seasonal_decompose(point_forecast, period=6, two_sided=True, extrapolate_trend=1).trend
    forecasted_trend_slope = (forecasted_trend.iloc[PERIOD_FOR_SLOPE-1] - forecasted_trend.iloc[0]) / PERIOD_FOR_SLOPE

    slope_criteria = ((forecasted_trend_slope - ts_trend_slope)/ts_trend_slope < FORECASTED_TREND_REDUCTION_THR) and\
                     (ts_trend_slope > 0)
    distinct_value_criteria = len(set(np.round(forecasted_trend.values))) < TREND_DISTINCT_VALUE_THR

    if slope_criteria or distinct_value_criteria:
        model = AutoTS(forecast_length=VALIDATION_POINTS,
                       no_negatives=NO_NEGATIVES,
                       ensemble=None,
                       max_generations=MAX_GENERATIONS,
                       model_list=MODEL_LIST,
                       verbose=VERBOSE,
                       num_validations=NUM_VALIDATIONS)
        model = model.fit(ts.reset_index(), date_col=DATE_COL, value_col=VALUE_COL, id_col=None)
        point_forecast = model.predict(FORECAST_PERIOD).forecast[VALUE_COL]

    return point_forecast, model


def noise_addition(ts: Series, point_forecast: Series) -> Series:
    res = seasonal_decompose(ts.interpolate(limit_direction='both'), period=6, two_sided=True,
                             extrapolate_trend=1).resid
    noise = np.random.normal(res.mean(), res.std(), len(point_forecast))
    output_values = point_forecast + noise * np.linspace(0, 1, len(point_forecast))

    return output_values
