import pandas as pd
from tsLQC.user_input import inputTableLocation, COMPANY_LIST
from tsLQC.preprocess_input import preprocessing
from tsLQC.execute import train_all_companies
from tsLQC.plot_forecast import plot_forecast


# input time series and preprocess it into the format needed
timeseries_input_df = pd.read_csv(inputTableLocation, index_col=0)
timeseries_input_df = preprocessing(timeseries_input_df)
timeseries_input_df = timeseries_input_df[timeseries_input_df.CompanyName.isin(COMPANY_LIST)]

if __name__ == '__main__':
    forecast_df = train_all_companies(timeseries_input_df)

    for i in timeseries_input_df.CompanyName.unique():
        try:
            plot_forecast(timeseries_input_df, forecast_df, i)
        except:
            pass
