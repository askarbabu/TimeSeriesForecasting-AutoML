import pandas as pd
from train_one_company import modelling
from forecasting import forecasting_function
from user_input import autots_hyperparameter_tuning

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
