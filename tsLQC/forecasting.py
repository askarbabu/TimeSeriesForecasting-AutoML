from user_input import models_with_custom_interval, forecast_period
from prediction_interval import prediction_interval


def forecasting_function(ts, model):
    model_pred = model.predict(forecast_period, prediction_interval=0.90)

    if model.best_model_name in models_with_custom_interval:
        point_forecast = model_pred.forecast
        lower_forecast, upper_forecast = prediction_interval(ts, point_forecast)
    else:
        point_forecast, lower_forecast, upper_forecast = model_pred.forecast, model_pred.lower_forecast, model_pred.upper_forecast

    return point_forecast, lower_forecast, upper_forecast
