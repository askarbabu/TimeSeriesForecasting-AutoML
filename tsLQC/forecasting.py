import logging
import pandas as pd
from autots import AutoTS
from typing import Tuple
from tsLQC.constant import forecast_period, confidence_interval, VALUE_COL
from tsLQC.prediction_interval import prediction_interval
from tsLQC.curve_flat import _revenue_flat
from tsLQC.postprocessing import bad_forecast_handling, noise_addition
import json


def forecasting_function(ts: pd.Series, model: AutoTS, best_models: pd.DataFrame) ->\
        Tuple[pd.Series, pd.Series, pd.Series]:
    point_forecast, lower_forecast, upper_forecast = pd.Series(), pd.Series(), pd.Series()

    for i in best_models.index:
        try:

            best_model_id = best_models.iloc[i]['ID']

            model.best_model_id = best_model_id
            model.best_model_name = best_models[best_models.ID == best_model_id]['Model'].iloc[0]
            model.best_model_params = \
                json.loads(best_models[best_models.ID == best_model_id]['ModelParameters'].iloc[0])
            model.best_model_transformation_params = \
                json.loads(best_models[best_models.ID == best_model_id]['TransformationParameters'].iloc[0])

            logging.info(f'Forecasting using {model.best_model_name} model')

            model_pred = model.predict(forecast_period, prediction_interval=confidence_interval)
            point_forecast = model_pred.forecast[VALUE_COL]

            point_forecast, model = bad_forecast_handling(ts, point_forecast, model)
            point_forecast = _revenue_flat(point_forecast)
            point_forecast = noise_addition(ts, point_forecast)

            lower_forecast, upper_forecast = prediction_interval(ts, point_forecast, model)
            point_forecast.name, lower_forecast.name, upper_forecast.name = 'point_forecast', 'lower_forecast', \
                                                                            'upper_forecast'
            break

        except:
            logging.info(f'Forecasting failed using {model.best_model_name} model')
            pass

    return point_forecast, lower_forecast, upper_forecast
