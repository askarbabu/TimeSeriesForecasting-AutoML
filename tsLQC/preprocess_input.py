import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose


PERCENT_TRIMMED = 0.1
STD_DEV_THR = 3


def outlier_treatment(ts: pd.Series) -> pd.Series:
    trimmed_len = int(PERCENT_TRIMMED*len(ts))

    res = seasonal_decompose(ts.interpolate(limit_direction='both').iloc[trimmed_len:-trimmed_len], period=6,
                             two_sided=True, extrapolate_trend=1).resid

    res_mean, res_std = res.mean(), res.std()
    outlier_upper_cutoff = res_mean + STD_DEV_THR * res.std()
    outlier_lower_cutoff = res_mean - STD_DEV_THR * res.std()

    outlier_idx = [idx + trimmed_len for idx in np.where((res > outlier_upper_cutoff) |
                                                         (res < outlier_lower_cutoff))[0]]
    ts.iloc[outlier_idx] = np.nan

    return ts
