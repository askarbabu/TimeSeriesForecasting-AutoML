import pandas as pd
from autots import AutoTS
from typing import Tuple
from tsLQC.autots_hyperparameter_tuning import hyperparameter_tuning
from tsLQC.forecasting import forecasting_function
from tsLQC.constant import frequency, no_negatives, n_jobs, ensemble, DATE_COL, VALUE_COL,\
    validation_method_default, validation_points_default, autots_hyperparameter_tuning, metric_weighting, \
    max_generations, num_validations, models_to_validate, model_list, verbose
from tsLQC.preprocess_input import outlier_treatment
from tsLQC.template_generation import template_generation, generate_ensemble_models
df = template_generation()


def modelling(ts: pd.Series, autots_hyperparameter_tuning: bool = False) -> Tuple[AutoTS, pd.DataFrame]:

    if autots_hyperparameter_tuning:
        validation_points, validation_method = hyperparameter_tuning(ts, 12)
    else:
        validation_points, validation_method = validation_points_default, validation_method_default

    model = AutoTS(forecast_length=validation_points,
                   frequency=frequency,
                   models_to_validate=models_to_validate,
                   no_negatives=no_negatives,
                   ensemble=ensemble,
                   max_generations=max_generations,
                   num_validations=num_validations,
                   validation_method=validation_method,
                   model_list=model_list,
                   n_jobs=n_jobs,
                   metric_weighting=metric_weighting,
                   verbose=verbose
                   )
    model = model.import_template(df, method='only')
    model = model.fit(ts.reset_index(), date_col=DATE_COL, value_col=VALUE_COL, id_col=None)
    best_models = model.export_template(models='best', n=100, max_per_model_class=None, include_results=True)\
        .sort_values('Score', ignore_index=True)

    return model, best_models


def train_one_company(ts: pd.Series) -> pd.DataFrame:
    try:
        ts = outlier_treatment(ts)
        model, best_simple_models = modelling(ts, autots_hyperparameter_tuning=autots_hyperparameter_tuning)
        model, best_models = generate_ensemble_models(ts, model, best_simple_models)
        point_forecast, lower_forecast, upper_forecast = forecasting_function(ts, model, best_models)
        forecast_one_company = pd.concat([point_forecast, lower_forecast, upper_forecast], axis=1)
        return forecast_one_company

    except:
        return pd.DataFrame()


def train_all_companies(timeseries_input_df: pd.DataFrame) -> dict:

    forecast_df = {i: train_one_company(ts=timeseries_input_df.loc[i].set_index(DATE_COL)[VALUE_COL])
                   for i in timeseries_input_df.index.unique()}

    return forecast_df
