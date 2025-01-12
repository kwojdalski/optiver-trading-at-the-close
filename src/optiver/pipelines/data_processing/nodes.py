import logging
from itertools import combinations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split

logger = logging.getLogger(__name__)


def save_train_data(train_path: str) -> pd.DataFrame:
    """Save training data"""
    train = pd.read_csv(train_path)
    train.to_pickle(train_path.replace(".csv", ".pkl"))


def load_train_data(
    train_data_path: str, frac: float = 1.0, random_state: int = 47
) -> pd.DataFrame:
    """Load and preprocess training data"""
    train = pd.read_pickle(train_data_path)
    train = train[~train["target"].isna()]

    train = train.sample(frac=frac, random_state=random_state)
    # Scale size columns
    size_col = ["imbalance_size", "matched_size", "bid_size", "ask_size"]
    for col in size_col:
        train[f"scale_{col}"] = train[col] / train.groupby(["stock_id"])[col].transform(
            "median"
        )

    # Calculate auction sizes
    train["auc_bid_size"] = train["matched_size"]
    train["auc_ask_size"] = train["matched_size"]
    train.loc[train["imbalance_buy_sell_flag"] == 1, "auc_bid_size"] += train.loc[
        train["imbalance_buy_sell_flag"] == 1, "imbalance_size"
    ]
    train.loc[train["imbalance_buy_sell_flag"] == -1, "auc_ask_size"] += train.loc[
        train["imbalance_buy_sell_flag"] == -1, "imbalance_size"
    ]

    return train


def generate_features(df: pd.DataFrame) -> tuple:
    """Generate features for training"""

    # Basic features
    df["ask_money"] = df["ask_size"] * df["ask_price"]
    df["bid_money"] = df["bid_size"] * df["bid_price"]
    df["ask_size_all"] = df["ask_size"] + df["auc_ask_size"]
    df["bid_size_all"] = df["bid_size"] + df["auc_bid_size"]
    df["volume_size_all"] = df["ask_size_all"] + df["bid_size_all"]
    df["ask_auc_money"] = df["reference_price"] * df["auc_ask_size"]
    df["bid_auc_money"] = df["reference_price"] * df["auc_bid_size"]
    df["volume_money"] = df["ask_money"] + df["bid_money"]
    df["volume_cont"] = df["ask_size"] + df["bid_size"]
    df["diff_ask_bid_size"] = df["ask_size"] - df["bid_size"]
    df["volume_auc"] = df["imbalance_size"] + 2 * df["matched_size"]
    df["volume_auc_money"] = df["volume_auc"] * df["reference_price"]
    df["mid_price"] = (df["ask_price"] + df["bid_price"]) / 2
    df["mid_price_near_far"] = (df["near_price"] + df["far_price"]) / 2
    df["price_diff_ask_bid"] = df["ask_price"] - df["bid_price"]
    df["price_div_ask_bid"] = df["ask_price"] / df["bid_price"]
    df["flag_scale_imbalance_size"] = (
        df["imbalance_buy_sell_flag"] * df["scale_imbalance_size"]
    )
    df["flag_imbalance_size"] = df["imbalance_buy_sell_flag"] * df["imbalance_size"]
    df["div_flag_imbalance_size_2_balance"] = (
        df["imbalance_size"] / df["matched_size"] * df["imbalance_buy_sell_flag"]
    )
    df["price_pressure"] = df["price_diff_ask_bid"] * df["imbalance_size"]
    df["price_pressure_v2"] = df["price_pressure"] * df["imbalance_buy_sell_flag"]
    df["depth_pressure"] = df["diff_ask_bid_size"] / (
        df["far_price"] - df["near_price"]
    )
    df["div_bid_size_ask_size"] = df["bid_size"] / df["ask_size"]

    # Ratio features
    ratio_pairs = [
        ("imbalance_size", "bid_size"),
        ("imbalance_size", "ask_size"),
        ("matched_size", "bid_size"),
        ("matched_size", "ask_size"),
        ("imbalance_size", "volume_cont"),
        ("matched_size", "volume_cont"),
        ("auc_bid_size", "bid_size"),
        ("auc_ask_size", "ask_size"),
        ("bid_auc_money", "bid_money"),
        ("ask_auc_money", "ask_money"),
    ]

    for col1, col2 in ratio_pairs:
        df[f"div_{col1}_2_{col2}"] = df[col1] / df[col2]

    # Imbalance features
    imb_pairs = [
        ("ask_size", "bid_size"),
        ("ask_money", "bid_money"),
        ("volume_money", "volume_auc_money"),
        ("volume_cont", "volume_auc"),
        ("imbalance_size", "matched_size"),
        ("auc_ask_size", "auc_bid_size"),
        ("ask_size_all", "bid_size_all"),
    ]

    for pair1, pair2 in imb_pairs:
        df[f"imb1_{pair1}_{pair2}"] = (df[pair1] - df[pair2]) / (df[pair1] + df[pair2])

    # Price imbalance features
    prices = [
        "reference_price",
        "far_price",
        "near_price",
        "ask_price",
        "bid_price",
        "wap",
        "mid_price",
    ]
    for c in combinations(prices, 2):
        df[f"imb1_{c[0]}_{c[1]}"] = (df[c[0]] - df[c[1]]) / (df[c[0]] + df[c[1]])

    # Market urgency features
    df["market_urgency_v2"] = (
        (df["imb1_ask_size_bid_size"] + 2)
        * (df["imb1_ask_price_bid_price"] + 2)
        * (df["imb1_auc_ask_size_auc_bid_size"] + 2)
    )
    df["market_urgency"] = df["price_diff_ask_bid"] * df["imb1_ask_size_bid_size"]
    df["market_urgency_v3"] = (
        df["imb1_ask_price_bid_price"] * df["imb1_ask_size_bid_size"]
    )

    # Rolling features
    for col in [
        "bid_auc_money",
        "imb1_reference_price_wap",
        "bid_size_all",
        "imb1_auc_ask_size_auc_bid_size",
        "div_flag_imbalance_size_2_balance",
        "imb1_ask_size_all_bid_size_all",
        "flag_imbalance_size",
        "imb1_reference_price_mid_price",
    ]:
        for window in [3, 6, 18, 36, 60]:
            df[f"rolling{window}_mean_{col}"] = df.groupby("stock_id")[col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f"rolling{window}_std_{col}"] = df.groupby("stock_id")[col].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )

    # Selected features list
    feas_list = [
        "imb1_wap_mid_price",
        "imb1_ask_money_bid_money",
        "imb1_volume_cont_volume_auc",
        "imb1_reference_price_ask_price",
        "imb1_reference_price_mid_price",
        "seconds_in_bucket",
        "div_flag_imbalance_size_2_balance",
        "ask_price",
        "imb1_reference_price_bid_price",
        "scale_matched_size",
        "imb1_near_price_wap",
        "volume_auc_money",
        "imb1_far_price_wap",
        "bid_size",
        "scale_bid_size",
        "bid_size_all",
    ]

    return df, feas_list


def train_model(df: pd.DataFrame, features: list, params: dict) -> dict:
    """Train XGBoost models with k-fold CV"""
    models = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=47)

    for k, (train_idx, test_idx) in enumerate(kf.split(df), 1):
        logging.info(f"Fold {k} begins...")

        train_data = df.iloc[train_idx]
        date_ids = train_data["date_id"].values
        weights = np.ones_like(date_ids, dtype=float)
        weights[date_ids >= 435] = 1.5

        logging.info(f"Training model for fold {k}...")

        model = xgb.XGBRegressor(**params)
        model.fit(
            train_data[features],
            train_data["target"],
            sample_weight=weights,
            eval_set=[(train_data[features], train_data["target"])],
            verbose=50,
        )

        logging.info(f"Model for fold {k} trained successfully.")

        models[f"fold_{k}"] = model

    return models


def split_data(
    data: pd.DataFrame, target: str, test_size: float, random_state: int
) -> tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """

    y = data[target]
    X = data.drop(target, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def train_ols_model(X_train: pd.DataFrame, y_train: pd.Series, params: dict) -> dict:
    """Train OLS models with k-fold CV"""
    models = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=47)

    for k, (train_idx, test_idx) in enumerate(kf.split(X_train), 1):
        logging.info(f"Fold {k} begins...")

        train_data = X_train.iloc[train_idx]
        date_ids = train_data["date_id"].values
        weights = np.ones_like(date_ids, dtype=float)
        weights[date_ids >= 435] = 1.5

        logging.info(f"Training model for fold {k}...")

        model = LinearRegression(**params)
        model.fit(train_data, y_train, sample_weight=weights)

        logging.info(f"Model for fold {k} trained successfully.")

        models[f"fold_{k}"] = model

    return models


def handle_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    return df


def check_and_replace_infinity_values(df: pd.DataFrame) -> pd.DataFrame:
    """Check for infinity values in training and test sets by column and replace with mean"""
    train_inf = df.isin([np.inf, -np.inf]).sum()

    logging.info("\nNumber of infinity values by column:")
    for col, count in train_inf[train_inf > 0].items():
        logging.info(f"{col}: {count}")
        # Replace inf values with mean for this column
        col_mean = df[col].replace([np.inf, -np.inf], np.nan).mean()
        df[col] = df[col].replace([np.inf, -np.inf], col_mean)

    if (train_inf > 0).any():
        logging.warning("Infinity values have been replaced with column means")

    return df
