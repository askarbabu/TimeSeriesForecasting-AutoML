import pandas as pd
from autots import AutoTS
from typing import Tuple
from tsLQC.autots_hyperparameter_tuning import hyperparameter_tuning
from tsLQC.forecasting import forecasting_function
from tsLQC.constant import FREQUENCY, NO_NEGATIVES, N_JOBS, DATE_COL, VALUE_COL,\
                           METRIC_WEIGHTING, MODELS_TO_VALIDATE, VERBOSE
from tsLQC.preprocess_input import outlier_treatment
from tsLQC.template_generation import template_generation, generate_ensemble_models


AUTOTS_HYPERPARAMETER_TUNING = True
MAX_GENERATIONS = 15
NUM_VALIDATIONS = 2
VALIDATION_POINTS_DEFAULT = 4
VALIDATION_METHOD_DEFAULT = 'backward'
MODEL_LIST = ['GLS', 'SeasonalNaive', 'GLM', 'ETS', 'WindowRegression', 'DatepartRegression',
              'UnivariateMotif', 'SectionalMotif', 'NVAR', 'ARIMA', 'ARDL', 'Theta']
ENSEMBLE = None
MAX_NO_OF_BEST_MODELS = 100

df = template_generation()


def modelling(ts: pd.Series, autots_hyperparameter_tuning: bool = AUTOTS_HYPERPARAMETER_TUNING) -> Tuple[AutoTS, pd.DataFrame]:

    if autots_hyperparameter_tuning:
        validation_points, validation_method = hyperparameter_tuning(ts, 12)
    else:
        validation_points, validation_method = VALIDATION_POINTS_DEFAULT, VALIDATION_METHOD_DEFAULT

    model = AutoTS(forecast_length=validation_points,
                   frequency=FREQUENCY,
                   models_to_validate=MODELS_TO_VALIDATE,
                   no_negatives=NO_NEGATIVES,
                   ensemble=ENSEMBLE,
                   max_generations=MAX_GENERATIONS,
                   num_validations=NUM_VALIDATIONS,
                   validation_method=validation_method,
                   model_list=MODEL_LIST,
                   n_jobs=N_JOBS,
                   metric_weighting=METRIC_WEIGHTING,
                   verbose=VERBOSE
                   )
    model = model.import_template(df, method='only')
    model = model.fit(ts.reset_index(), date_col=DATE_COL, value_col=VALUE_COL, id_col=None)
    best_models = model.export_template(models='best', n=MAX_NO_OF_BEST_MODELS, max_per_model_class=None,
                                        include_results=True).sort_values('Score', ignore_index=True)

    return model, best_models


def train_one_company(ts: pd.Series) -> pd.DataFrame:
    try:
        ts = outlier_treatment(ts)
        model, best_simple_models = modelling(ts, autots_hyperparameter_tuning=AUTOTS_HYPERPARAMETER_TUNING)
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
