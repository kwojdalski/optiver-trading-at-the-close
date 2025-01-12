from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    check_and_replace_infinity_values,
    generate_features,
    handle_nan_rows,
    load_train_data,
    split_data,
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
                func=split_data,
                inputs=[
                    "processed_train_data_no_nan",
                    "params:target",
                    "params:test_size",
                    "params:random_state",
                ],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
        ]
    )
