import pandas as pd
from autots import AutoTS
from tsLQC.autots_hyperparameter_tuning import hyperparameter_tuning
from tsLQC.forecasting import forecasting_function
from tsLQC.user_input import autots_hyperparameter_tuning, metric_weighting, \
    max_generations, num_validations, models_to_validate

from template_generation import templateGeneration
df = templateGeneration()


def modelling(ts, autots_hyperparameter_tuning=False):

    if autots_hyperparameter_tuning:
        validation_points, validation_method = hyperparameter_tuning(ts, 12)
    else:
        validation_points, validation_method = 4, 'backward'

    model_list = [i for i in df.Model.unique() if (i != 'FBProphet')]
    model = AutoTS(forecast_length=validation_points,
                   frequency='infer',
                   models_to_validate=models_to_validate,
                   no_negatives=True,
                   ensemble='simple',
                   max_generations=max_generations,
                   num_validations=num_validations,
                   validation_method=validation_method,
                   model_list=model_list,
                   n_jobs=7,
                   metric_weighting=metric_weighting,
                   )
    model = model.import_template(df, method='only')
    model = model.fit(ts.reset_index(), date_col='Date', value_col='Value', id_col=None)

    return model


def train_all_companies(timeseries_input_df):
    forecast_df = pd.DataFrame({'CompanyID': [], 'CompanyName': [], 'Date': [],
                                'PointForecast': [], 'LowerForecast': [], 'UpperForecast': []})

    for i in timeseries_input_df.index.unique():

        try:
            ts = timeseries_input_df.loc[i].set_index('Date')[['Value']]

            company_id = i
            company_name = timeseries_input_df[timeseries_input_df.index == i]['CompanyName'].iloc[0]

            print('*********************Time Series Modelling for ' + company_name + '*****************************')

            model = modelling(ts, autots_hyperparameter_tuning=autots_hyperparameter_tuning)
            point_forecast, lower_forecast, upper_forecast = forecasting_function(ts, model)

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
