from tsLQC.constant import models_with_custom_interval, forecast_period, confidence_interval
from tsLQC.prediction_interval import prediction_interval
from tsLQC.curve_flat import _revenue_flat


def forecasting_function(ts, model):
    model_pred = model.predict(forecast_period, prediction_interval=confidence_interval)

    point_forecast = model_pred.forecast
    point_forecast = _revenue_flat(point_forecast)
    if model.best_model_name in models_with_custom_interval:
        lower_forecast, upper_forecast = prediction_interval(ts, point_forecast, model)
    else:
        lower_forecast, upper_forecast = model_pred.lower_forecast, model_pred.upper_forecast

    return point_forecast, lower_forecast, upper_forecast
