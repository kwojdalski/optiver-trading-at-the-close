# %% [markdown]
# # Overview
# In this competition, you are challenged to develop a model capable of predicting the closing price
# movements for hundreds of Nasdaq listed stocks using data from the order book and the closing auction of
# the stock. Information from the auction can be used to adjust prices, access supply and demand dynamics,
# and identify trading opportunities.
#
# # Description
#
# #### [Taken from the competition website](https://www.kaggle.com/competitions/optiver-trading-at-the-close/overview)
# Stock exchanges are fast-paced, high-stakes environments where every second counts. The intensity
# escalates as the trading day approaches its end, peaking in the critical final ten minutes. These
# moments, often characterised by heightened volatility and rapid price fluctuations, play a pivotal role
# in shaping the global economic narrative for the day.
#
# Each trading day on the Nasdaq Stock Exchange concludes with the Nasdaq Closing Cross auction. This
# process establishes the official closing prices for securities listed on the exchange. These closing
# prices serve as key indicators for investors, analysts and other market participants in evaluating the
# performance of individual securities and the market as a whole.
#
# Within this complex financial landscape operates Optiver, a leading global electronic market maker.
# Fueled by technological innovation, Optiver trades a vast array of financial instruments, such as
# derivatives, cash equities, ETFs, bonds, and foreign currencies, offering competitive, two-sided prices
# for thousands of these instruments on major exchanges worldwide.
#
# In the last ten minutes of the Nasdaq exchange trading session, market makers like Optiver merge
# traditional order book data with auction book data. This ability to consolidate information from both
# sources is critical for providing the best prices to all market participants.
#
# In this competition, you are challenged to develop a model capable of predicting the closing price
# movements for hundreds of Nasdaq listed stocks using data from the order book and the closing auction
# of the stock. Information from the auction can be used to adjust prices, assess supply and demand
# dynamics, and identify trading opportunities.
#
# Your model can contribute to the consolidation of signals from the auction and order book, leading to
# improved market efficiency and accessibility, particularly during the intense final ten minutes of
# trading. You'll also get firsthand experience in handling real-world data science problems, similar to
# those faced by traders, quantitative researchers and engineers at Optiver.
#
# # Evaluation
# Submissions are evaluate on the Mean Absolute Error (MAE) between the predicted return and the observed target. The formula is given by:
#
# $$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - x_i|$$
#
# Where:
#
# - $n$ is the total number of data points.
# - $y_i$ is the predicted value for data point $i$.
# - $x_i$ is the observed value for data point $i$.
#

# ## Data Description
# %% [markdown]
# Dataset Description
# This dataset contains historic data for the daily ten minute closing auction on the NASDAQ stock
# exchange. Your challenge is to predict the future price movements of stocks relative to the price future
# price movement of a synthetic index composed of NASDAQ-listed stocks.
#
# This is a forecasting competition using the time series API. The private leaderboard will be determined
# using real market data gathered after the submission period closes.
#
# Files
# [train/test].csv The auction data. The test data will be delivered by the API.
#
# * stock_id - A unique identifier for the stock. Not all stock IDs exist in every time bucket.
# * date_id - A unique identifier for the date. Date IDs are sequential & consistent across all stocks.
# * imbalance_size - The amount unmatched at the current reference price (in USD).
# * imbalance_buy_sell_flag - An indicator reflecting the direction of auction imbalance:
#   - buy-side imbalance: 1
#   - sell-side imbalance: -1
#   - no imbalance: 0
# * reference_price - Price where shares are maximized, imbalance minimized, and distance from mid minimized
# * matched_size - The amount that can be matched at the current reference price (in USD).
# * far_price - Crossing price maximizing matched shares based on auction interest only.
# * near_price - Crossing price maximizing matched shares based on auction and continuous market orders.
# [bid/ask]_price - Price of the most competitive buy/sell level in the non-auction book.
# [bid/ask]_size - Dollar notional amount on the most competitive buy/sell level.
# wap - Weighted average price in the non-auction book:
#   $$\frac{BidPrice \times AskSize + AskPrice \times BidSize}{BidSize + AskSize}$$
# seconds_in_bucket - Seconds elapsed since start of closing auction, from 0.
# target - 60-second future move in stock WAP minus 60-second future move of synthetic index.
#
# The synthetic index is a custom weighted index of Nasdaq stocks constructed by Optiver.
# Target is in basis points (1bp = 0.01%). Formula:
# $$Target = \left(\frac{StockWAP_{t+60}}{StockWAP_t} - \frac{IndexWAP_{t+60}}{IndexWAP_t}\right) \times 10000$$
#
# All size columns are in USD.
# All price columns are moves relative to the stock WAP at auction start.
#
# #### [Taken from the competition website](https://www.kaggle.com/competitions/optiver-trading-at-the-close/overview)


# %%
import logging
from multiprocessing import context

import pandas as pd
from IPython import get_ipython
from kedro.ipython import load_ipython_extension
from plotnine import aes, facet_wrap, geom_histogram, ggplot, theme

from src.optiver.pipelines.data_processing.nodes import train_ols_model

ipython = get_ipython()
load_ipython_extension(ipython)
sc = logging.getLogger(__name__)
sc.setLevel(logging.INFO)

params = context.params
# %%
# Loading data
# Moreover the function adds variables based on matched size
out0 = (
    pipelines["data_processing"]
    .nodes[0]
    .run(
        {
            "train_data_path": "data/01_raw/train.pkl",
            "params:raw_data_frac": params["raw_data_frac"],
            "params:random_state": params["random_state"],
        }
    )
)

# %%


def plot_all_histograms(data: pd.DataFrame):
    """
    Create histograms for all numeric variables in the dataset

    Args:
        data: Input dataframe containing numeric columns
    """
    # Sample 10% of data with fixed random seed
    sampled_data = data.sample(frac=0.00001, random_state=0)

    # Melt the dataframe to get all numeric columns in long format
    numeric_cols = sampled_data.select_dtypes(include=["int64", "float64"]).columns
    melted_data = pd.melt(sampled_data[numeric_cols])

    # Create histogram for each numeric variable
    plot = (
        ggplot(melted_data, aes(x="value"))
        + geom_histogram()
        + facet_wrap("~variable", scales="free")
        + theme(figure_size=(15, 15))
    )

    return plot


# Generate and display the plots
# histogram_plot = plot_all_histograms(data)
# %% [markdown]
# Check for missing values in the dataset
# * Looks like there are no missing values for most columns in the dataset
# * The only columns with missing values are `far_price` (55.26%) and `near_price` (54.55%)
# %%
data = out0["raw_train_data"]
sc.info("\nMissing values in each column:")
sc.info(data.isna().sum())
sc.info("\nMissing values percentage in each column:")
sc.info((data.isna().sum() / len(data) * 100).round(2))

# Display basic dataset information
sc.info("\nDataset Description:")
sc.info(f"Number of rows: {len(data)}")

sc.info(f"Number of columns: {len(data.columns)}")
sc.info("\nColumn descriptions:")
sc.info(data.describe())

# Display data types of columns
sc.info("\nData types of columns:")
sc.info(data.dtypes)

# %%
out1 = (
    pipelines["data_processing"]
    .nodes[1]
    .run({"raw_train_data": out0["raw_train_data"]})
)
processed_train_data = out1["processed_train_data"]
feature_list = out1["feature_list"]

# %%
# handling missing values / outliers / uncalculated values
out2 = (
    pipelines["data_processing"]
    .nodes[2]
    .run(
        {
            "processed_train_data": processed_train_data,
        }
    )
)

processed_train_data_no_nan = out2["processed_train_data_no_nan"]
# %%
out3 = (
    pipelines["data_processing"]
    .nodes[3]
    .run(
        {
            "processed_train_data_no_nan": processed_train_data_no_nan,
        }
    )
)
# %%
out4 = (
    pipelines["data_processing"]
    .nodes[4]
    .run(
        {
            "processed_train_data_no_nan": processed_train_data_no_nan,
            "params:test_size": params["test_size"],
            "params:random_state": params["random_state"],
            "params:target": params["target"],
        }
    )
)

X_train, X_test, y_train, y_test = (
    out4["X_train"],
    out4["X_test"],
    out4["y_train"],
    out4["y_test"],
)


# %%

train_ols_model(X_train, y_train, params={})


# Check for infinity values in training and test sets


check_infinity_values(X_train, X_test)
# Remove rows with infinity values
X_train = X_train.replace([np.inf, -np.inf], 0).dropna()
X_test = X_test.replace([np.inf, -np.inf], 0).dropna()

# Update y_train and y_test to match filtered X
y_train = y_train[X_train.index]
y_test = y_test[X_test.index]

# Key findings
# * Infinity values are present in the dataset
