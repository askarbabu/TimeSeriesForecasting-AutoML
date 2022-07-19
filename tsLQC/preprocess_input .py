import pandas as pd
from tsLQC.user_input import primary_key


def preprocessing(timeseries_input_df):

    timeseries_input_df['Date'] = timeseries_input_df['Date'].str[-4:] + '-' + timeseries_input_df['Date'].str[
                                                                               3:5] + '-01'
    timeseries_input_df['Date'] = pd.to_datetime(timeseries_input_df['Date'])
    timeseries_input_df['CompanyClass'] = timeseries_input_df['CompanyClass'].astype('str')
    timeseries_input_df = timeseries_input_df.set_index(primary_key)
    timeseries_input_df.index = timeseries_input_df.index.astype(int).astype(str)

    return timeseries_input_df

print(primary_key)
