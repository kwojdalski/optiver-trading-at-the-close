import logging

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    coord_equal,
    element_text,
    facet_wrap,
    geom_boxplot,
    geom_histogram,
    geom_line,
    geom_point,
    geom_smooth,
    geom_tile,
    ggplot,
    labs,
    scale_fill_gradient2,
    theme,
)


# %%
def plot_market_data(df: pd.DataFrame, stock_id: int = 0, date_id: int = 0) -> None:
    """Plot various market data visualizations for a given stock and date

    Args:
        df: DataFrame containing stock market data
        stock_id: ID of stock to plot (default 0)
        date_id: ID of date to plot (default 0)
    """
    # Plot bid/ask/wap prices
    plot_df = (
        df[(df["stock_id"] == stock_id) & (df["date_id"] == date_id)][
            ["seconds_in_bucket", "bid_price", "ask_price", "wap"]
        ]
        .replace(0, np.nan)
        .melt(
            id_vars=["seconds_in_bucket"],
            value_vars=["bid_price", "ask_price", "wap"],
            var_name="price_type",
            value_name="price",
        )
    )

    logging.info(f"Plotting data for stock {stock_id} on day {date_id}")
    # Create and display bid/ask/wap plot
    bid_ask_wap_plot = (
        ggplot(plot_df, aes(x="seconds_in_bucket", y="price", color="price_type"))
        + geom_line()
        + labs(
            x="Seconds in Bucket",
            y="Price",
            title=f"Bid, Ask and WAP Prices for Stock {stock_id} on Day {date_id}",
            color="Price Type",
        )
        + theme(figure_size=(12, 6))
    )

    # Plot imbalance and matched size
    plot_df = df[(df["stock_id"] == stock_id) & (df["date_id"] == date_id)][
        ["seconds_in_bucket", "imbalance_size", "matched_size"]
    ].melt(
        id_vars=["seconds_in_bucket"],
        value_vars=["imbalance_size", "matched_size"],
        var_name="size_type",
        value_name="size",
    )

    imbalance_matched_plot = (
        ggplot(plot_df, aes(x="seconds_in_bucket", y="size", color="size_type"))
        + geom_line()
        + labs(
            x="Seconds in Bucket",
            y="Size",
            title=f"Imbalance and Matched Sizes for Stock {stock_id} on Day {date_id}",
            color="Size Type",
        )
        + theme(figure_size=(12, 6))
    )
    # Plot near/far/reference prices
    plot_df = (
        df[(df["stock_id"] == stock_id) & (df["date_id"] == date_id)][
            ["seconds_in_bucket", "near_price", "far_price", "reference_price"]
        ]
        .replace(0, np.nan)
        .melt(
            id_vars=["seconds_in_bucket"],
            value_vars=["near_price", "far_price", "reference_price"],
            var_name="price_type",
            value_name="price",
        )
    )

    price_plot = (
        ggplot(plot_df, aes(x="seconds_in_bucket", y="price", color="price_type"))
        + geom_line()
        + labs(
            x="Seconds in Bucket",
            y="Price",
            title=f"Stock {stock_id} on Day {date_id} - Near, far and reference prices",
            color="Price Type",
        )
        + theme(figure_size=(12, 6))
    )

    return bid_ask_wap_plot, imbalance_matched_plot, price_plot


def plot_time_series(df: pd.DataFrame, date_id: int):
    """Plot time series for target in day 0 for each stock

    Args:
        df: DataFrame containing stock data with columns stock_id, target,
            seconds_in_bucket and date_id
    """
    df_plt = df.query(f"date_id == {date_id}")[
        ["stock_id", "target", "seconds_in_bucket"]
    ]

    return (
        ggplot(df_plt, aes(x="seconds_in_bucket", y="target", color="factor(stock_id)"))
        + geom_line()
        + labs(
            x="Seconds in Bucket",
            y="Target",
            title="Multiple Time Series for Each Stock",
        )
        + theme(legend_position="none")
    )


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


# %%
def plot_correlation_matrix(df):
    """
    Create and plot a correlation matrix heatmap for the given dataframe.

    Args:
        df (pd.DataFrame): Input dataframe to compute correlations for

    Returns:
        None: Displays the correlation heatmap plot
    """
    # compute the correlation matrix
    corr_matrix = df.corr()

    # Convert correlation matrix to long format for plotnine
    corr_df = corr_matrix.reset_index().melt(id_vars="index")
    corr_df.columns = ["Var1", "Var2", "value"]

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
    corr_df = corr_df[~mask.ravel()]

    # Create heatmap with plotnine
    p = (
        ggplot(corr_df, aes("Var1", "Var2", fill="value"))
        + geom_tile()
        + scale_fill_gradient2(
            low="brown", mid="white", high="green", limits=[-0.35, 0.35], midpoint=0
        )
        + theme(
            figure_size=(12, 10),
            axis_text_x=element_text(rotation=90, size=9),
            axis_text_y=element_text(size=9),
        )
        + coord_equal()
        + labs(title="Correlation Matrix")
    )
    print(p)


def plot_feature_distributions(data: pd.DataFrame, normalize: bool = False) -> None:
    """Create box plots to visualize feature distributions.

    Args:
        data: DataFrame containing features to plot
        normalize: bool, whether to normalize features (default=False)
    """
    numeric_features = [
        col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])
    ]
    data = data[numeric_features]

    if normalize:
        # Normalize numeric features excluding stock_id and date_id
        features_to_normalize = [
            col for col in data.columns if col not in ["stock_id", "date_id"]
        ]
        data[features_to_normalize] = (
            data[features_to_normalize] - data[features_to_normalize].mean()
        ) / data[features_to_normalize].std()

    plot_df = data.melt(
        id_vars=["stock_id", "date_id"],
        value_vars=[col for col in data.columns if col not in ["stock_id", "date_id"]],
    )

    box_plot = (
        ggplot(plot_df, aes(x="variable", y="value"))
        + geom_boxplot()
        + theme(axis_text_x=element_text(rotation=45, hjust=1), figure_size=(15, 8))
        + labs(title="Distribution of Features", x="Features", y="Value")
    )

    return box_plot


def plot_feature_relationships(
    df: pd.DataFrame, feature_columns: list, target_column: str = "target"
):
    """Create faceted scatter plots showing relationships between features and target.

    Args:
        df: DataFrame containing features and target
        feature_columns: List of feature column names to plot
        target_column: Name of target column (default: "target")
    """
    # Create a long format dataframe for plotting
    # Filter out non-numeric columns to avoid dtype errors
    numeric_features = [
        col for col in feature_columns if pd.api.types.is_numeric_dtype(df[col])
    ]
    df = df[numeric_features + [target_column]]
    dff = df.loc[:, ~df.columns.duplicated()].copy()
    plot_data = pd.melt(
        dff,
        id_vars=[target_column],
        var_name="feature",
        value_name="value",
    )

    # Create faceted scatter plot with trend lines
    plot = (
        ggplot(plot_data, aes(x="value", y=target_column))
        + geom_point(alpha=0.5)
        + geom_smooth(method="lm", color="red")
        + facet_wrap("~feature", ncol=4, scales="free_x")
        + theme(figure_size=(25, 25))
        + labs(x="", y="Target")
    )
    return plot


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
