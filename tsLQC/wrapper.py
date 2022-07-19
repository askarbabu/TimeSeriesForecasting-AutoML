import os
print(os.getcwd())

from user_input import inputTableLocation
print(inputTableLocation)

# from tsLQC.preprocess_new import preprocessing
# from preprocess_new import preprocessing
import pandas as pd

timeseries_input_df = pd.read_csv(inputTableLocation, index_col=0)
timeseries_input_df = preprocessing(timeseries_input_df)
print(timeseries_input_df.shape)