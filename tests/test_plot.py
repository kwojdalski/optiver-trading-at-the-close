# %%
import pandas as pd
import pytest
from plotnine import ggplot

from src.optiver.plots import (
    plot_correlation_matrix,
    plot_feature_distributions,
    plot_feature_relationships,
    plot_market_data,
    plot_time_series,
)


@pytest.fixture
def sample_data():
    """Create sample DataFrame for testing plots."""
    return pd.DataFrame(
        {
            "stock_id": [1, 1, 2, 2],
            "date_id": [1, 1, 1, 1],
            "seconds_in_bucket": [0, 1, 0, 1],
            "target": [0.1, 0.2, -0.1, -0.2],
            "imbalance_size": [100, 200, 150, 250],
            "matched_size": [90, 180, 140, 230],
            "reference_price": [10.0, 10.1, 9.9, 9.8],
            "far_price": [10.1, 10.2, 9.8, 9.7],
            "near_price": [10.0, 10.1, 9.9, 9.8],
            "bid_price": [9.9, 10.0, 9.8, 9.7],
            "ask_price": [10.1, 10.2, 10.0, 9.9],
            "wap": [10.0, 10.1, 9.9, 9.8],
            "bid_size": [50, 60, 45, 55],
            "ask_size": [40, 50, 35, 45],
        }
    )


def sample_data2():
    """Create sample DataFrame for testing plots."""
    return pd.DataFrame(
        {
            "stock_id": [1, 1, 2, 2],
            "date_id": [1, 1, 1, 1],
            "seconds_in_bucket": [0, 1, 0, 1],
            "target": [0.1, 0.2, -0.1, -0.2],
            "imbalance_size": [100, 200, 150, 250],
            "matched_size": [90, 180, 140, 230],
            "reference_price": [10.0, 10.1, 9.9, 9.8],
            "far_price": [10.1, 10.2, 9.8, 9.7],
            "near_price": [10.0, 10.1, 9.9, 9.8],
            "bid_price": [9.9, 10.0, 9.8, 9.7],
            "ask_price": [10.1, 10.2, 10.0, 9.9],
            "wap": [10.0, 10.1, 9.9, 9.8],
            "bid_size": [50, 60, 45, 55],
            "ask_size": [40, 50, 35, 45],
        }
    )


def test_plot_market_data(sample_data):
    """Test market data plotting function returns three ggplot objects."""
    plots = plot_market_data(sample_data)
    assert len(plots) == 3
    for plot in plots:
        assert isinstance(plot, ggplot)


def test_plot_time_series(sample_data):
    """Test time series plotting function returns a ggplot object."""
    plot = plot_time_series(sample_data, date_id=1)
    assert isinstance(plot, ggplot)


def test_plot_feature_distributions(sample_data):
    """Test feature distributions plotting function returns a ggplot object."""
    plot = plot_feature_distributions(sample_data)
    assert isinstance(plot, ggplot)


def test_plot_correlation_matrix(sample_data):
    """Test correlation matrix plotting function executes without error."""
    try:
        plot_correlation_matrix(sample_data)
    except Exception as e:
        pytest.fail(f"plot_correlation_matrix raised an exception: {e}")


def test_plot_feature_relationships(sample_data):
    """Test feature relationships plotting function executes without error."""
    feature_cols = [
        "imbalance_size",
        "matched_size",
        "reference_price",
        "bid_size",
        "ask_size",
    ]
    try:
        plot_feature_relationships(sample_data, feature_cols, "target")
    except Exception as e:
        pytest.fail(f"plot_feature_relationships raised an exception: {e}")


# %%
df = sample_data2()
plot_feature_relationships(df, df.columns.tolist(), "target")
# plot_correlation_matrix(df)
plot_feature_distributions(df)
# plot_time_series(df)
# plot_market_data(df)
# %%
from plotnine import aes, facet_wrap, geom_point, geom_smooth, labs, theme

# Filter numeric features
numeric_features = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
# Remove any duplicated columns
numeric_features = list(dict.fromkeys(numeric_features))
dff = df[numeric_features + ["target"]]

# Create long format data
plot_data = pd.melt(dff, id_vars=["target"], var_name="feature", value_name="value")

# Create and display faceted scatter plot
plot = (
    ggplot(plot_data, aes(x="value", y="target"))
    + geom_point(alpha=0.5)
    + geom_smooth(method="lm", color="red")
    + facet_wrap("~feature", ncol=4, scales="free_x")
    + theme(figure_size=(30, 30))
    + labs(x="", y="Target")
)

plot.draw()
