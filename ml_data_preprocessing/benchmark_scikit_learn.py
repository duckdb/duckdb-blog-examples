import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from utils import get_data_as_pd, timeit


@timeit
def scikit_encode(data_df):
    encoding_steps = ColumnTransformer(
        [
            (
                "onehot",
                OneHotEncoder(),
                ["payment_channel", "device_used", "merchant_category"],
            ),
            ("ordinal", OrdinalEncoder(), ["transaction_type"]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    encoded_data = encoding_steps.fit_transform(data_df)

    return pd.DataFrame(
        encoded_data,
        columns=encoding_steps.get_feature_names_out(),
        index=data_df.index,
    )


@timeit
def scikit_split_data(encoded_data_df):
    x_train, x_test = train_test_split(encoded_data_df, test_size=0.2, random_state=256)
    return x_train, x_test


@timeit
def scikit_feature_scaling_training_data(x_train):
    impute_missing_data = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", MinMaxScaler(copy=False)),
        ]
    )

    scaling_steps = ColumnTransformer(
        [
            ("ss", StandardScaler(copy=False), ["velocity_score"]),
            (
                "minmax_time_since_last_transaction",
                impute_missing_data,
                ["time_since_last_transaction"],
            ),
            ("minmax", MinMaxScaler(copy=False), ["spending_deviation_score"]),
            ("rs", RobustScaler(copy=False), ["amount"]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    scaling_steps.set_output(transform="pandas")
    scaling_steps.fit(x_train)

    return scaling_steps, scaling_steps.transform(x_train)


@timeit
def scikit_feature_scaling_testing_data(scaling_steps, x_test):
    return scaling_steps.transform(x_test)


@timeit
def main():
    raw_data_df = get_data_as_pd()

    for i in range(1, 2):
        print(f"Iteration {i}")
        encoded_df = scikit_encode(raw_data_df)
        x_train_df, x_test_df = scikit_split_data(encoded_df)
        scaling_steps_proc, x_train_transformed_df = (
            scikit_feature_scaling_training_data(x_train_df)
        )
        x_test_transformed_df = scikit_feature_scaling_testing_data(
            scaling_steps_proc, x_test_df
        )


if __name__ == "__main__":
    main()
