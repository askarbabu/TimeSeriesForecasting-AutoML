import pandas as pd
from autots import AutoTS
from tsLQC.autots_hyperparameter_tuning import hyperparameter_tuning
from tsLQC.forecasting import forecasting_function
from tsLQC.constant import frequency, no_negatives, n_jobs, ensemble, date_col, value_col,\
    validation_method_default, validation_points_default, autots_hyperparameter_tuning, metric_weighting, \
    max_generations, num_validations, models_to_validate
from tsLQC.preprocess_input import outlier_treatment

from tsLQC.template_generation import template_generation, generate_ensemble_models
df = template_generation()


def modelling(ts, autots_hyperparameter_tuning=False):

    if autots_hyperparameter_tuning:
        validation_points, validation_method = hyperparameter_tuning(ts, 12)
    else:
        validation_points, validation_method = validation_points_default, validation_method_default

    model_list = [i for i in df.Model.unique() if (i != 'FBProphet')]
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
                   )
    model = model.import_template(df, method='only')
    model = model.fit(ts.reset_index(), date_col=date_col, value_col=value_col, id_col=None)
    best_models = model.export_template(models='best', n=100, max_per_model_class=None, include_results=True) \
        .sort_values('Score', ignore_index=True)

    return model, best_models


def train_all_companies(timeseries_input_df):
    forecast_df = pd.DataFrame({'CompanyID': [], 'CompanyName': [], 'Date': [],
                                'PointForecast': [], 'LowerForecast': [], 'UpperForecast': []})

    for i in timeseries_input_df.index.unique():

        try:
            ts = timeseries_input_df.loc[i].set_index('Date')[['Value']]
            ts = outlier_treatment(ts)

            company_id = i
            company_name = timeseries_input_df[timeseries_input_df.index == i]['CompanyName'].iloc[0]

            print('*********************Time Series Modelling for ' + company_name + '*****************************')

            model, best_simple_models = modelling(ts, autots_hyperparameter_tuning=autots_hyperparameter_tuning)
            model, best_models = generate_ensemble_models(ts, model, best_simple_models)
            point_forecast, lower_forecast, upper_forecast = forecasting_function(ts, model, best_models)

            temp_df = pd.DataFrame({'CompanyID': [str(company_id)] * len(point_forecast),
                                    'CompanyName': [company_name] * len(point_forecast),
                                    'Date': list(point_forecast.index),
                                    'PointForecast': list(point_forecast['Value'].values),
                                    'LowerForecast': list(lower_forecast['Value'].values),
                                    'UpperForecast': list(upper_forecast['Value'].values)
                                    })
            forecast_df = pd.concat([forecast_df, temp_df])

        except:
            print(
                '##################################### FAILED FOR {} ######################################'.format(i))

    forecast_df = forecast_df.set_index('CompanyID')

    return forecast_df
