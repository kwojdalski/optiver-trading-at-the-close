from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    calculate_feature_importance,
    check_and_replace_infinity_values,
    generate_features,
    handle_nan_rows,
    load_train_data,
    remove_outliers,
    split_data,
    train_knn_model,
    train_linear_model,
    train_svr_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the training pipeline"""
    return pipeline(
        [
            node(
                func=load_train_data,
                inputs=[
                    "train_data_path",
                    "params:raw_data_frac",
                    "params:random_state",
                ],
                outputs="raw_train_data",
                name="load_train_data",
            ),
            node(
                func=generate_features,
                inputs="raw_train_data",
                outputs=["processed_train_data", "feature_list"],
                name="generate_features",
            ),
            node(
                func=handle_nan_rows,
                inputs="processed_train_data",
                outputs="processed_train_data_no_nan",
                name="handle_nan_rows",
            ),
            node(
                func=check_and_replace_infinity_values,
                inputs="processed_train_data_no_nan",
                outputs="processed_train_data_no_nan_no_inf",
                name="check_and_replace_infinity_values",
            ),
            node(
                func=calculate_feature_importance,
                inputs=[
                    "processed_train_data_no_nan_no_inf",
                    "params:all_features",
                ],
                outputs=["filtered_feat_data", "general_ranking", "columns_to_drop"],
                name="calculate_feature_importance",
            ),
            node(
                func=remove_outliers,
                inputs="filtered_feat_data",
                outputs="filtered_feat_data_no_outliers",
                name="remove_outliers",
            ),
            node(
                func=split_data,
                inputs=[
                    "filtered_feat_data_no_outliers",
                    "params:target",
                    "params:test_size",
                    "params:random_state",
                ],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_linear_model,
                inputs=["X_train", "X_test", "y_train", "y_test"],
                outputs="lr_model",
                name="train_linear_model",
            ),
            node(
                func=train_svr_model,
                inputs=["X_train", "X_test", "y_train", "y_test"],
                outputs="svr_model",
                name="train_svr_model",
            ),
            node(
                func=train_knn_model,
                inputs=["X_train", "X_test", "y_train", "y_test"],
                outputs="knn_model",
                name="train_knn_model",
            ),
        ]
    )
