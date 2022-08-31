import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tsLQC.preprocess_input import preprocessing
from tsLQC.execute import train_all_companies
from tsLQC.constant import DATE_COL, VALUE_COL

INPUT_TABLE_LOCATION = 'revenue_input_v3.csv'
COMPANY_LIST = ['ExeVir', 'Franklin', 'Optiqua', 'Micropharma']

# input time series and preprocess it into the format needed
timeseries_input_df = pd.read_csv(INPUT_TABLE_LOCATION, index_col=0)
timeseries_input_df = preprocessing(timeseries_input_df)
timeseries_input_df = timeseries_input_df[timeseries_input_df.CompanyName.isin(COMPANY_LIST)]


def plot_forecast(timeseries_input_df: pd.DataFrame, forecast_df: dict, plot_company: str) -> None:
    df = timeseries_input_df.loc[plot_company]
    company_name = df.iloc[0]['CompanyName']
    historical_data = df.set_index(DATE_COL)[VALUE_COL]

    point_forecast = forecast_df[plot_company]['point_forecast']
    upper_forecast = forecast_df[plot_company]['upper_forecast']
    lower_forecast = forecast_df[plot_company]['lower_forecast']

    sns.set_style("whitegrid")

    fig, ax1 = plt.subplots(1, 1, sharex='all', figsize=(8, 12))
    ax1.plot(historical_data, label='Historical Data', linestyle='dashed', c='k', marker='o')
    ax1.plot(point_forecast, label='Forecast', c='#0343DF')
    ax1.fill_between(point_forecast.index, lower_forecast, upper_forecast, alpha=0.2)
    ax1.set_xlim([historical_data.index[0], point_forecast.index[-1]])
    ax1.set_ylabel('Revenue in USD')
    ax1.set_xlabel('Date')
    ax1.set_title('Forecast for ' + company_name)
    plt.show()

    return


if __name__ == '__main__':
    forecast_df = train_all_companies(timeseries_input_df)

    for i in timeseries_input_df.index.unique():
        try:
            plot_forecast(timeseries_input_df, forecast_df, i)
        except:
            pass
