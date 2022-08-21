from tsLQC.constant import models_with_custom_interval, forecast_period, confidence_interval
from tsLQC.prediction_interval import prediction_interval
from tsLQC.curve_flat import _revenue_flat
from tsLQC.postprocessing import bad_forecast_handling, noise_addition
import json


def forecasting_function(ts, model, best_models):
    try:

        model = pick_best_model(model, best_models)
        print('Forecasting using', model.best_model_name, 'model')

        model_pred = model.predict(forecast_period, prediction_interval=confidence_interval)
        point_forecast = model_pred.forecast

        point_forecast, model = bad_forecast_handling(ts, point_forecast, model)
        point_forecast = _revenue_flat(point_forecast)
        point_forecast = noise_addition(ts, point_forecast)

        if model.best_model_name in models_with_custom_interval:
            lower_forecast, upper_forecast = prediction_interval(ts, point_forecast, model)
        else:
            lower_forecast, upper_forecast = model_pred.lower_forecast, model_pred.upper_forecast

        return point_forecast, lower_forecast, upper_forecast

    except:
        print('Forecasting failed using', model.best_model_name, 'model')

        next_best_models = best_models.shift(-1)

        return forecasting_function(ts, model, next_best_models)


def pick_best_model(model, best_models):
    best_model_id = best_models.iloc[0]['ID']

    model.best_model_id = best_model_id
    model.best_model_name = best_models[best_models.ID == best_model_id]['Model'].iloc[0]
    model.best_model_params = json.loads(best_models[best_models.ID == best_model_id]['ModelParameters'].iloc[0])
    model.best_model_transformation_params = \
        json.loads(best_models[best_models.ID == best_model_id]['TransformationParameters'].iloc[0])

    return model
