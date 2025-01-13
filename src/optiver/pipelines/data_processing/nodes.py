import logging
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

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


def train_linear_model(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> dict:
    """Train linear regression models with k-fold CV and evaluate performance,
    merging coefficients across folds

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target

    Returns:
        Tuple containing merged model and test score
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=47)
    params = {"fit_intercept": True}

    # Lists to store coefficients and intercepts from each fold
    all_coef = []
    all_intercepts = []

    for k, (train_idx, test_idx) in enumerate(kf.split(X_train), 1):
        logging.info(f"Fold {k} begins...")

        train_data = X_train.iloc[train_idx]
        y_train_fold = y_train.iloc[train_idx]

        logging.info(f"Training model for fold {k}...")
        logging.info(f"Train data shape: {train_data.shape}")
        logging.info(f"Y train shape: {y_train_fold.shape}")

        model = LinearRegression(**params)
        model.fit(train_data, y_train_fold)

        all_coef.append(model.coef_)
        all_intercepts.append(model.intercept_)

        logging.info(f"Model for fold {k} trained successfully.")

    # Create merged model with averaged coefficients
    final_model = LinearRegression(**params)
    final_model.coef_ = np.mean(all_coef, axis=0)
    final_model.intercept_ = np.mean(all_intercepts)

    # Make predictions with merged model
    test_pred = final_model.predict(X_test)

    # Calculate test performance metrics
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    mae = mean_absolute_error(y_test, test_pred)
    r2 = r2_score(y_test, test_pred)

    logging.info("\nTest Set Metrics:")
    logging.info(f"RMSE: {rmse:.6f}")
    logging.info(f"MAE: {mae:.6f}")
    logging.info(f"R2 Score: {r2:.6f}")

    metrics = {"rmse": rmse, "mae": mae, "r2": r2}

    return final_model, metrics


def train_knn_model(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> tuple:
    """Train KNN models with k-fold CV and evaluate performance

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target

    Returns:
        Tuple containing final model and test score
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=47)
    params = {"n_neighbors": 5}

    # Lists to store predictions from each fold
    all_predictions = []

    for k, (train_idx, test_idx) in enumerate(kf.split(X_train), 1):
        logging.info(f"Fold {k} begins...")

        train_data = X_train.iloc[train_idx]
        y_train_fold = y_train.iloc[train_idx]

        logging.info(f"Training model for fold {k}...")
        logging.info(f"Train data shape: {train_data.shape}")
        logging.info(f"Y train shape: {y_train_fold.shape}")
        model = KNeighborsRegressor(**params)
        model.fit(train_data, y_train_fold)

        logging.info(f"Model for fold {k} trained successfully.")

        # Get predictions for this fold
        fold_pred = model.predict(X_test)
        all_predictions.append(fold_pred)

    # Train final model on full training data
    final_model = KNeighborsRegressor(**params)
    final_model.fit(X_train, y_train)

    # Average predictions across folds
    test_pred_avg = np.mean(all_predictions, axis=0)

    # Calculate test performance metrics
    rmse = np.sqrt(mean_squared_error(y_test, test_pred_avg))
    mae = mean_absolute_error(y_test, test_pred_avg)
    r2 = r2_score(y_test, test_pred_avg)

    logging.info("\nTest Set Metrics:")
    logging.info(f"RMSE: {rmse:.6f}")
    logging.info(f"MAE: {mae:.6f}")
    logging.info(f"R2 Score: {r2:.6f}")

    metrics = {"rmse": rmse, "mae": mae, "r2": r2}

    return final_model, metrics


def train_svr_model(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> tuple:
    """Train SVM models with k-fold CV and evaluate performance

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target

    Returns:
        Tuple containing final model and test score
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=47)
    params = {"kernel": "rbf", "C": 1.0, "epsilon": 0.1}

    # Lists to store predictions from each fold
    all_predictions = []

    for k, (train_idx, test_idx) in enumerate(kf.split(X_train), 1):
        logging.info(f"Fold {k} begins...")

        train_data = X_train.iloc[train_idx]
        y_train_fold = y_train.iloc[train_idx]

        logging.info(f"Training model for fold {k}...")
        logging.info(f"Train data shape: {train_data.shape}")
        logging.info(f"Y train shape: {y_train_fold.shape}")
        model = SVR(**params)
        model.fit(train_data, y_train_fold)

        logging.info(f"Model for fold {k} trained successfully.")

        # Get predictions for this fold
        fold_pred = model.predict(X_test)
        all_predictions.append(fold_pred)

    # Train final model on full training data
    final_model = SVR(**params)
    final_model.fit(X_train, y_train)

    # Average predictions across folds
    test_pred_avg = np.mean(all_predictions, axis=0)

    # Calculate test performance metrics
    rmse = np.sqrt(mean_squared_error(y_test, test_pred_avg))
    mae = mean_absolute_error(y_test, test_pred_avg)
    r2 = r2_score(y_test, test_pred_avg)

    logging.info("\nTest Set Metrics:")
    logging.info(f"RMSE: {rmse:.6f}")
    logging.info(f"MAE: {mae:.6f}")
    logging.info(f"R2 Score: {r2:.6f}")

    metrics = {"rmse": rmse, "mae": mae, "r2": r2}

    return final_model, metrics


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
