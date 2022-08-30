from datetime import date
from typing import List

import numpy as np
from pandas import Series
from scipy.optimize import curve_fit

from tsLQC.constant import flattening_analysis_range, flattening_limit_multiplier


def _revenue_flat(revenue: Series) -> Series:
    flt_rev = TimeSeriesFlattener(revenue)
    flat_rev = flt_rev.flatten()
    return flat_rev


class TimeSeriesFlattener:
    def __init__(self, ts: Series):
        self.ts = ts

    def flatten(self) -> Series:
        predicted_data = self.ts
        start_flattening_date = self._find_start_flattening_index(predicted_data)

        if (start_flattening_date is None) or start_flattening_date >= predicted_data.index[-1]:
            return self.ts
        flattened_ts = self._apply_flat(predicted_data, start_flattening_date)
        return flattened_ts

    @staticmethod
    def _apply_flat(ts: Series, start_flattening_date) -> Series:
        def _expv_func(x, a, b):
            return a / (1 + np.exp(-b * x))

        start_flattening_position = ts.index.get_loc(start_flattening_date)

        step_delta = int(flattening_analysis_range * start_flattening_position)
        values2fit = ts.values[(start_flattening_position - step_delta):(start_flattening_position + step_delta)]
        if values2fit.size > 0:
            [p1, p2], params_covariance = curve_fit(f=_expv_func,
                                                    xdata=np.arange(len(values2fit)),
                                                    ydata=values2fit,
                                                    p0=[values2fit[0], 1e-2],
                                                    maxfev=5000)
            flat_trend = _expv_func(np.arange(len(ts)), p1, p2)
            ind2switch = np.argmin(np.abs(flat_trend[:step_delta] - values2fit[:step_delta]))
            fixed_values = flat_trend[ind2switch:].tolist()
            result_values = ts[:start_flattening_position - step_delta + ind2switch].tolist() + fixed_values
            output_values = result_values[:len(ts)]
            return Series(output_values, index=ts.index)
        return ts

    @staticmethod
    def _add_noise(result_values: List[float]) -> List[float]:
        amplitude = np.max(np.diff(result_values)) / 6
        noise = np.random.normal(-amplitude, amplitude, len(result_values))
        return result_values + noise * np.linspace(0, 1, len(result_values))

    @staticmethod
    def _find_start_flattening_index(ts: Series) -> date:
        ref = ts[:12].median()
        return ts.loc[ts > flattening_limit_multiplier * ref].first_valid_index()
