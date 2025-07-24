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
    )

    encoded_data = encoding_steps.fit_transform(data_df)

    return pd.DataFrame(
        encoded_data,
        columns=[
            name.replace("remainder__", "")
            for name in encoding_steps.get_feature_names_out()
        ],
        index=data_df.index,
    )


@timeit
def scikit_split_data(encoded_data_df):
    x_train, x_test = train_test_split(encoded_data_df, test_size=0.2, random_state=256)
    return x_train, x_test


@timeit
def scikit_get_transformed_training_data(x_train):
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
    )

    scaling_steps.fit(x_train)

    return scaling_steps, pd.DataFrame(
        scaling_steps.transform(x_train),
        columns=[
            name.replace("remainder__", "")
            for name in scaling_steps.get_feature_names_out()
        ],
        index=x_train.index,
    )


@timeit
def scikit_get_transformed_testing_data(scaling_steps, x_test):
    return pd.DataFrame(
        scaling_steps.transform(x_test),
        columns=[
            name.replace("remainder__", "")
            for name in scaling_steps.get_feature_names_out()
        ],
        index=x_test.index,
    )


@timeit
def main():
    raw_data_df = get_data_as_pd()

    for i in range(1, 11):
        print(f"Iteration {i}")
        encoded_df = scikit_encode(raw_data_df)
        x_train_df, x_test_df = scikit_split_data(encoded_df)
        scaling_steps_proc, x_train_transformed_df = (
            scikit_get_transformed_training_data(x_train_df)
        )
        x_test_transformed_df = scikit_get_transformed_testing_data(
            scaling_steps_proc, x_test_df
        )


if __name__ == "__main__":
    main()
