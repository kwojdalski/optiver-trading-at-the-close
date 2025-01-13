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
import warnings
from multiprocessing import context

import numpy as np
import pandas as pd
from IPython import get_ipython
from kedro.ipython import load_ipython_extension
from plotnine import aes, facet_wrap, geom_point, geom_smooth, ggplot, labs, theme
from sklearn.feature_selection import (
    VarianceThreshold,
    f_regression,
    mutual_info_regression,
)

from src.optiver.plots import (
    plot_correlation_matrix,
    plot_feature_distributions,
    plot_feature_relationships,
    plot_market_data,
    plot_time_series,
)

ipython = get_ipython()
load_ipython_extension(ipython)
sc = logging.getLogger(__name__)
sc.setLevel(logging.INFO)

params = context.params
initial_features = params["features"]
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
# Ploting time series for target in day 0

processed_train_data = catalog.load("raw_train_data")

processed_train_data[processed_train_data["date_id"] == 2]
plot_time_series(processed_train_data, date_id=2)


plots = plot_market_data(processed_train_data)
plots[0].show()
plots[1].show()
plots[2].show()

# Generate and display the plots
# histogram_plot = plot_all_histograms(data)
# %% [markdown]
# Inpsect the dataset
# * The dataset consists of only numeric features that can be used in the model
#     + row_id, stock_id, time_id are used as identifiers for the final submission
#     + target is the target variable
#     + all other columns are used as features
#     + moreover, we can construct more features that might be used for further alpha extraction
#         + the whole point of this exercise is to predict the closing price (given by the specific formula from description)
#         + if we can predict that alpha accurately, we can use it to trade the stock
# .       + for instance, by posting bids below / offers on non-primaries above expected value (e.g. TURQ vs LSE)
# * Looks like there are no missing values for most columns in the dataset
#    + the only columns with missing values are `far_price` (55.26%) and `near_price` (54.55%)

# %%
data = out0["raw_train_data"]

# Display data types of columns
# only numeric, no categorical features
sc.info("\nData types of columns:")
sc.info(data.dtypes)
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


# %%
# Check for correlation between features
# There is a strong correlation between some of the features
# Hence, additional test for collinearity is needed
# Create box plots to visualize feature distributions
# %% plot feature distributions
# Initial feature distributions vary by magnitude and scale. Hence, further
# normalization might be required
plot_feature_distributions(data, normalize=True)

# %% [markdown] Feature relationships visualization
# * each feature is plotted against the target variable
# * we can clearly see outliers in the data
plot_feature_relationships(data.sample(frac=0.01), data.columns.tolist(), "target")

# %%
# Plot correlation matrix
# * there is a strong correlation between some of the features
#     + in some cases, in comes from the fact that one feature is a linear combination of two others
# (like `scale_bid_size` and `scale_ask_size` vs `bid_size` and `ask_size`)
# * Additional test for collinearity might be useful
plot_correlation_matrix(data)

## %%
# Perform collinearity test using Variance Inflation Factor (VIF)


# %% [markdown]
# Feature engineering
# This node generates features for the model
# %% [markdown] Feature engineering formulas
# Generate features for training with formulas in LaTeX notation:
#
# ### Basic Features:
# - Ask Money: $$AM = AS \times AP$$
# - Bid Money: $$BM = BS \times BP$$
# - Ask Size All: $$ASA = AS + AAS$$
# - Bid Size All: $$BSA = BS + ABS$$
# - Volume Size All: $$VSA = ASA + BSA$$
# - Ask Auction Money: $$AAM = RP \times AAS$$
# - Bid Auction Money: $$BAM = RP \times ABS$$
# - Volume Money: $$VM = AM + BM$$
# - Volume Continuous: $$VC = AS + BS$$
# - Diff Ask Bid Size: $$DABS = AS - BS$$
# - Volume Auction: $$VA = IS + 2MS$$
# - Volume Auction Money: $$VAM = VA \times RP$$
# - Mid Price: $$MP = \frac{AP + BP}{2}$$
# - Mid Price Near Far: $$MPNF = \frac{NP + FP}{2}$$
# - Price Diff Ask Bid: $$PDAB = AP - BP$$
# - Price Div Ask Bid: $$PDAB = \frac{AP}{BP}$$
# - Flag Scale Imbalance Size: $$FSIS = IBSF \times SIS$$
# - Flag Imbalance Size: $$FIS = IBSF \times IS$$
# - Div Flag Imbalance Size to Balance: $$DFISB = \frac{IS}{MS} \times IBSF$$
# - Price Pressure: $$PP = PDAB \times IS$$
# - Price Pressure v2: $$PPv2 = PP \times IBSF$$
# - Depth Pressure: $$DP = \frac{DABS}{FP - NP}$$
# - Div Bid Size Ask Size: $$DBSAS = \frac{BS}{AS}$$
#
# ### Ratio Features:
# For any pair $(X,Y)$ in ratio pairs:
# $$div\\_X\\_Y = \frac{X}{Y}$$
#
# ### Imbalance Features:
# For any pair $(X,Y)$ in imbalance pairs:
# $$imb1\\_X\\_Y = \frac{X-Y}{X+Y}$$
# Price Imbalance Features:
# For any pair of prices $(P1,P2)$:
# $imb1\_P1\_P2 = \frac{P1-P2}{P1+P2}$
#
# ### Market Urgency Features:
# - Market Urgency v2: $$MUv2 = (IASBS + 2)(IAPBP + 2)(IAASABS + 2)$$
# - Market Urgency: $$MU = PDAB \times IASBS$$
# - Market Urgency v3: $$MUv3 = IAPBP \times IASBS$$
#
# ### Rolling Features:
# For each feature $F$ and window size $w$:
# - Rolling Mean: $$\bar{F_w} = \frac{1}{w}\sum_{i=t-w+1}^t F_i$$
# - Rolling Std: $$\sigma_{F_w} = \sqrt{\frac{1}{w}\sum_{i=t-w+1}^t (F_i - \bar{F_w})^2}$$
#
# Where:
# * $AS$ = ask_size,
# * $AP$ = ask_price,
# * $BS$ = bid_size,
# * $BP$ = bid_price
# * $AAS$ = auc_ask_size,
# * $ABS$ = auc_bid_size,
# * $RP$ = reference_price
# * $IS$ = imbalance_size,
# * $MS$ = matched_size,
# * $NP$ = near_price
# * $FP$ = far_price,
# * $IBSF$ = imbalance_buy_sell_flag
# * $SIS$ = scale_imbalance_size
# * $IASBS$ = imb1_ask_size_bid_size
# * $IAPBP$ = imb1_ask_price_bid_price
# * $IAASABS$ = imb1_auc_ask_size_auc_bid_size

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
# %% [markdown]
# Feature engineering
# This node generates features for the model
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

# %% [markdown]
# Feature selection


# %% Feature selection
all_features = initial_features + feature_list
# Remove submission features from all features
all_features = [f for f in all_features if f not in params["submission_features"]]
proc_df = out3["processed_train_data_no_nan_no_inf"]
proc_df[all_features]

# %% [markdown] Feature selection
# * Variance thresholding helps identifying and removing features that show minimal variation across observations.
# * When a feature remains mostly constant or changes very little, it typically doesn't contribute meaningful
# predictive power.
# * Moreover, I use mi_score and sign_fscore to identify features with weak relationships with the target variable.
# Features with low mi_score ($mi\_score < 0.01$) and high sign_fscore ($sign\_fscore > 0.1$)
# should be removed as they have weak relationships with the target variable.


# %%
def calculate_feature_importance(
    proc_df: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Calculate feature importance scores using mutual information and f-regression.

    Args:
        proc_df: Processed dataframe containing features and target
        features: List of all feature names

    Returns:
        Tuple containing:
        - DataFrame with mutual information and f-regression scores for each feature
        - List of columns that should be dropped due to low importance
    """
    # Calculate mutual information scores
    sel = VarianceThreshold(0.01)
    sel_var = sel.fit_transform(proc_df[all_features])
    col_imp = proc_df[all_features][
        proc_df[all_features].columns[sel.get_support(indices=True)]
    ].columns
    col_redundant = set(processed_train_data[all_features].columns.tolist()) - set(
        col_imp
    )

    mi: dict[str, float] = dict()
    for feature in col_imp:
        mi.update(
            {
                feature: mutual_info_regression(
                    proc_df[[feature]].values, proc_df["target"].values
                )[0]
            }
        )
    miDF = pd.DataFrame.from_dict(mi, orient="index", columns=["score"])
    general_ranking = pd.DataFrame(index=all_features)
    general_ranking = pd.merge(general_ranking, miDF, left_index=True, right_index=True)
    general_ranking.rename(columns={"score": "mi_score"}, inplace=True)

    # Calculate f-regression scores
    warnings.simplefilter(action="ignore", category=FutureWarning)
    fscore: dict[str, float] = dict()
    for i in all_features:
        fscore.update(
            {i: f_regression(proc_df[[i]].values, proc_df["target"].values)[1]}
        )
    fscoreDF = pd.DataFrame.from_dict(fscore, orient="index", columns=["p_value_score"])
    fscoreDF.sort_values(by="p_value_score").head(10)
    fscoreDF.sort_values(by="p_value_score", ascending=False).head(10)
    fscoreDF["sign"] = np.where(fscoreDF.p_value_score < 0.1, 1, 0)
    general_ranking = pd.merge(
        general_ranking, fscoreDF, left_index=True, right_index=True
    )
    general_ranking.rename(
        columns={"p_value_score": "sign_fscore", "sign": "sign_fscore_0_1"},
        inplace=True,
    )

    # Identify columns to drop based on importance thresholds
    columns_to_drop = general_ranking[
        (general_ranking["mi_score"] < 0.01) & (general_ranking["sign_fscore"] > 0.1)
    ].index.tolist()

    return general_ranking, columns_to_drop


general_ranking, columns_to_drop = calculate_feature_importance(proc_df, all_features)


# %% [markdown]
# Split the data into train and test
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

# %% [markdown]
# Train Linear Regression models with k-fold CV
out5 = (
    pipelines["data_processing"]
    .nodes[5]
    .run(
        {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }
    )
)

# %% [markdown]
# Train Random Forest models with k-fold CV
# %%
out6 = (
    pipelines["data_processing"]
    .nodes[6]
    .run(
        {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }
    )
)

# %% [markdown]
# Train KNN models with k-fold CV
out8 = (
    pipelines["data_processing"]
    .nodes[7]
    .run(
        {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }
    )
)
# %%
# Merge all models into a single dictionary
models = {
    "knn": out5["knn_model"][0],  # Linear regression models
    "lr": out6["lr_model"][0],  # Random forest models
    # 'svr': out7[0],  # SVR models
    "svr": out8["svr_model"][0],  # KNN models
}

accuracy = {
    "knn": out5["knn_model"][1],  # Linear regression accuracy
    "lr": out6["lr_model"][1],  # Random forest accuracy
    # 'svr': out7[1],  # SVR accuracy
    "svr": out8["svr_model"][1],  # KNN accuracy
}


# %% plot the models
# Create predictions for plotting
predictions_dict = {}
for name, model in models.items():
    predictions_dict[f"{name}_pred"] = model.predict(X_test)


def plot_model_predictions(y_test: pd.Series, predictions_dict: dict):
    """Create scatter plots comparing model predictions vs actual values.

    Args:
        y_test: Series containing actual test values
        predictions_dict: Dictionary mapping model names to their predictions

    Returns:
        ggplot object with faceted scatter plots
    """
    # Create dataframe with actual values and predictions from each model
    plot_df = pd.DataFrame({"Actual": y_test, **predictions_dict})

    # Melt the dataframe to long format for faceting
    plot_df_long = pd.melt(
        plot_df,
        id_vars=["Actual"],
        value_vars=[col for col in plot_df.columns if col != "Actual"],
        var_name="Model",
        value_name="Predicted",
    )

    # Create scatter plots for each model
    plot = (
        ggplot(plot_df_long, aes(x="Actual", y="Predicted"))
        + geom_point(alpha=0.5)
        + geom_smooth(method="lm", color="red")
        + facet_wrap("~Model", ncol=3)
        + labs(title="Model Predictions vs Actual Values")
        + theme(figure_size=(25, 25))
    )

    return plot


# %% [markdown] Accuracy table
# * Optiver benchmark (MAE):
#     + baseline accuracy - 6.4077
#     + simple prediction accuracy - 6.4070 (MAE improvement in basis points: 0.0007)
# * We can that all of our models are slightly better with linear regression being the best one (5.7562)
accuracy_df = pd.DataFrame(accuracy).T
accuracy_df = accuracy_df.round(4)

# %%
# Key takewaways:
# * In practice, such models might be used for trading, but not for HFT, uHFT, as they tend to be too slow
# * The model is not able to predict the closing price with high accuracy
# * More sophisticated models might be needed to achieve better results that could contribute to alpha generation for MFT
# * Accuracy of models could be improved by working on parametrization of both models and engineered features
#     + Experiment with regularization methods
#     + Trying different models
# * Also, data processing pipeline could have been more sophisticated with addessing such issues as:
#     + cleaning up the data (e.g. outlier detection)
#     + feature selection
#     + handling missing values / inf values differently
#     + data transformation (e.g. normalization, log transformation, etc.)
#     + dimensionality reduction methods (e.g. PCA) that could speed up the training process at a relatively low cost (of accuracy)

# %%
# References:
# * [Optiver Trading at the Close Competition](https://www.optiver.com/en/)
# * [Github Repo used for this submission](https://github.com/kwojdalski/optiver-trading-at-the-close)
