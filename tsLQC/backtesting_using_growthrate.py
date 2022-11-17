import logging
from pandas import Series
from tsLQC.constant import FORECAST_PERIOD, DATE_COL, VALUE_COL
from autots import AutoTS
from typing import Tuple


FILL_VALUE = 0.1
PERIOD = 12


def compute_growth(x1: float, x2: float) -> float:
    return (x2 - x1) / x1


def rate_periodized(growthrate: float, period: int) -> float:
    if growthrate > -1:
        return round((((1 + growthrate) ** (1 / period)) - 1), 4)
    else:
        return -1


def backtesting(ts: Series, backtest_length: int, model: AutoTS) -> Tuple[float, float]:

    backtest_df = ts[-backtest_length:]
    temp_df = ts[:-backtest_length]

    baseline = temp_df[-backtest_length:].sum()
    cumulative_actual = backtest_df.sum()
    benchmark = rate_periodized(compute_growth(baseline, cumulative_actual), PERIOD)

    logging.info(f'benchmark: {benchmark}')

    temp_df[temp_df <= 0] = FILL_VALUE
    model = model.fit(temp_df.reset_index(), date_col=DATE_COL, value_col=VALUE_COL, id_col=None)
    prediction = model.predict(FORECAST_PERIOD)
    forecast_autots = prediction.forecast[VALUE_COL]

    cumulative_forecasted_autots = forecast_autots.iloc[:backtest_length].sum()
    forecasted_gr = rate_periodized(compute_growth(baseline, cumulative_forecasted_autots), PERIOD)

    logging.info(f'forecasted growth rate: {forecasted_gr}')

    return benchmark, forecasted_gr
