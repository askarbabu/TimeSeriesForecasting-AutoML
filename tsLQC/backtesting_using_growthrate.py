def compute_growth(x1, x2):
    return (x2 - x1) / x1


def rate_periodized(growthrate, period):
    if growthrate[0] > -1:
        return round((((1 + growthrate) ** (1 / period)) - 1), 4)
    else:
        return -1


def backtesting(ts, backtest_length, model):

    backtest_df = ts[-backtest_length:]
    temp_df = ts[:-backtest_length]

    baseline = temp_df[-backtest_length:].sum()
    cumulative_actual = backtest_df.sum()
    benchmark = rate_periodized(compute_growth(baseline, cumulative_actual), 12)[0]

    print("benchmark: ", benchmark)

    temp_df[temp_df['Value'] <= 0] = 0.1
    model = model.fit(temp_df.reset_index(), date_col='Date', value_col='Value', id_col=None)
    prediction = model.predict(48)
    forecast_autots = prediction.forecast

    cumulative_forecasted_autots = forecast_autots.iloc[:backtest_length].sum()
    forecasted_gr = rate_periodized(compute_growth(baseline, cumulative_forecasted_autots), 12)[0]

    print("forecasted growth rate: ", forecasted_gr)

    return benchmark, forecasted_gr
