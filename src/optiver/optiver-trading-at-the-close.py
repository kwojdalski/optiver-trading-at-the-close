# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" papermill={"duration": 0.937589, "end_time": "2024-03-07T01:59:55.316260", "exception": false, "start_time": "2024-03-07T01:59:54.378671", "status": "completed"}
#loading the necessary packages


import numpy as np # for linear data
import pandas as pd # data processing, csv files i/o



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



# %% [markdown] papermill={"duration": 0.008773, "end_time": "2024-03-07T01:59:55.334812", "exception": false, "start_time": "2024-03-07T01:59:55.326039", "status": "completed"}
#
#   
#
# wap = ( BidPrice∗AskSize + AskPrice∗BidSize) / (BidSize+AskSize)
#
# target = ( (StockWAP[t+60] / StockWAP [t]) − (IndexWAP [t+60] IndexWAP [t]) )∗10000
#
# - columns:
#     1. date_id = days
#     2. seconds_in_bucket = final ::: 10 minutes -> 600 seconds
#     

# %% papermill={"duration": 0.018151, "end_time": "2024-03-07T01:59:55.361956", "exception": false, "start_time": "2024-03-07T01:59:55.343805", "status": "completed"}
## Looking at data

# %% papermill={"duration": 23.421896, "end_time": "2024-03-07T02:00:18.792928", "exception": false, "start_time": "2024-03-07T01:59:55.371032", "status": "completed"}
df = pd.read_csv('/kaggle/input/optiver-trading-at-the-close/train.csv')

# %% [markdown] papermill={"duration": 0.008635, "end_time": "2024-03-07T02:00:18.810898", "exception": false, "start_time": "2024-03-07T02:00:18.802263", "status": "completed"}
# # 1.Data Analysis

# %% papermill={"duration": 1.492837, "end_time": "2024-03-07T02:00:20.312661", "exception": false, "start_time": "2024-03-07T02:00:18.819824", "status": "completed"}
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go

# %% papermill={"duration": 0.908624, "end_time": "2024-03-07T02:00:21.230490", "exception": false, "start_time": "2024-03-07T02:00:20.321866", "status": "completed"}
# Ploting time series for target in day 0 

df_plt = (
    df
    .query('date_id == 0')
    [['stock_id', 'target', 'seconds_in_bucket']]
)

plt.figure(figsize=(10, 6))

for stock in df_plt['stock_id'].unique():
    df_stock = df_plt[df_plt['stock_id'] == stock]
    plt.plot(df_stock['seconds_in_bucket'], df_stock['target'], label=f'Stock {stock}')

plt.xlabel('Seconds in Bucket')
plt.ylabel('Target')
plt.title('Multiple Time Series for Each Stock')
plt.show()

# %% papermill={"duration": 0.497544, "end_time": "2024-03-07T02:00:24.173924", "exception": false, "start_time": "2024-03-07T02:00:23.676380", "status": "completed"}
# Ploting the bid and ask price to stock and day 0

(
    df
    .query('stock_id == 0 & date_id == 0')
    [['seconds_in_bucket','bid_price','ask_price', 'wap']]
    .replace(0, np.nan)
    .set_index('seconds_in_bucket')
    .plot(title = 'Bid and Ask over the stock and day 0')
)

# %% papermill={"duration": 0.483727, "end_time": "2024-03-07T02:00:24.675066", "exception": false, "start_time": "2024-03-07T02:00:24.191339", "status": "completed"}
# Ploting the near, far and reference prices

(
    df
    .query('stock_id ==0 & date_id ==0')
    [['seconds_in_bucket','near_price','far_price','reference_price']]
    .replace(0, np.nan)
    .set_index('seconds_in_bucket')
    .plot(title = 'Stock 0 on Day 0 - Ploting the near, far and reference prices')
)

# %% papermill={"duration": 0.54143, "end_time": "2024-03-07T02:00:25.235130", "exception": false, "start_time": "2024-03-07T02:00:24.693700", "status": "completed"}
# Imbalance and matched size plotting

(
    df
    .query('stock_id == 3 & date_id ==0')
    [['seconds_in_bucket','imbalance_size','matched_size']]
    .set_index('seconds_in_bucket')
    .plot(title='Stock 0 on Day 0 - Imbalance and matched')
)


# %% [markdown] papermill={"duration": 0.019583, "end_time": "2024-03-07T02:00:25.274500", "exception": false, "start_time": "2024-03-07T02:00:25.254917", "status": "completed"}
# # Feature engineering

# %% papermill={"duration": 0.030485, "end_time": "2024-03-07T02:00:25.324317", "exception": false, "start_time": "2024-03-07T02:00:25.293832", "status": "completed"}
def feature_cols(df):
    columns = [col for col in df.columns if col not in ['row_id', 'time_id', 'date_id', 'stock_id', 'currently_scored']]
    return df[columns]


# %% papermill={"duration": 1.702651, "end_time": "2024-03-07T02:00:27.046190", "exception": false, "start_time": "2024-03-07T02:00:25.343539", "status": "completed"}
df.fillna(0, inplace=True)

# %% papermill={"duration": 0.552003, "end_time": "2024-03-07T02:00:27.617917", "exception": false, "start_time": "2024-03-07T02:00:27.065914", "status": "completed"}
x_train = feature_cols(df.drop(columns='target'))
x_train.fillna(0, inplace = True)

# %% papermill={"duration": 0.029822, "end_time": "2024-03-07T02:00:27.667382", "exception": false, "start_time": "2024-03-07T02:00:27.637560", "status": "completed"}
y_train = df['target'].values

# %% [markdown] papermill={"duration": 0.019114, "end_time": "2024-03-07T02:00:27.706220", "exception": false, "start_time": "2024-03-07T02:00:27.687106", "status": "completed"}
# # Splitting the data

# %% papermill={"duration": 0.311017, "end_time": "2024-03-07T02:00:28.037198", "exception": false, "start_time": "2024-03-07T02:00:27.726181", "status": "completed"}
from sklearn.model_selection import train_test_split

# %% papermill={"duration": 1.876761, "end_time": "2024-03-07T02:00:29.934265", "exception": false, "start_time": "2024-03-07T02:00:28.057504", "status": "completed"}
X_train, x_test, Y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# %% [markdown] papermill={"duration": 0.020446, "end_time": "2024-03-07T02:00:29.974082", "exception": false, "start_time": "2024-03-07T02:00:29.953636", "status": "completed"}
# # Modelling

# %% papermill={"duration": 0.031244, "end_time": "2024-03-07T02:00:30.027668", "exception": false, "start_time": "2024-03-07T02:00:29.996424", "status": "completed"}
# Created the modeling with lgbm regressors 

# %% papermill={"duration": 1.36059, "end_time": "2024-03-07T02:00:31.407967", "exception": false, "start_time": "2024-03-07T02:00:30.047377", "status": "completed"}
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

import lightgbm as lgb
from lightgbm import LGBMRegressor

pd.set_option("display.max_columns", None)

# %% papermill={"duration": 0.033369, "end_time": "2024-03-07T02:00:31.460581", "exception": false, "start_time": "2024-03-07T02:00:31.427212", "status": "completed"}
from sklearn.pipeline import Pipeline, FunctionTransformer

model = lgb.LGBMRegressor(objective='mae', n_estimators=200, random_state=55)

pipe = Pipeline([
    ('select_features', FunctionTransformer(feature_cols, validate=False)),
    ('model', model)
])

# %% papermill={"duration": 66.937938, "end_time": "2024-03-07T02:01:38.417732", "exception": false, "start_time": "2024-03-07T02:00:31.479794", "status": "completed"}
pipe.fit(X_train, Y_train)

# %% papermill={"duration": 4.820508, "end_time": "2024-03-07T02:01:43.782824", "exception": false, "start_time": "2024-03-07T02:01:38.962316", "status": "completed"}
## Validating the model

y_pred = pipe.predict(x_test)

# %% papermill={"duration": 0.044018, "end_time": "2024-03-07T02:01:43.851813", "exception": false, "start_time": "2024-03-07T02:01:43.807795", "status": "completed"}
mae = mean_absolute_error(y_test, y_pred)

print(mae)
