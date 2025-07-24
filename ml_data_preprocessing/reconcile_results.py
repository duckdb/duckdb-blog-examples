import duckdb

from benchmark_duckdb import (
    duckdb_create_macros,
    duckdb_save_pd_as_table,
    duckdb_encode,
    duckdb_get_transformed_testing_data,
    duckdb_get_transformed_training_data,
)
from benchmark_scikit_learn import (
    scikit_encode,
    scikit_split_data,
    scikit_get_transformed_training_data,
    scikit_get_transformed_testing_data,
)
from utils import get_data_as_pd

raw_data_df = get_data_as_pd()

conn = duckdb.connect()
duckdb_create_macros(conn)

duckdb_save_pd_as_table(conn)

print("Reconcile encoders")
duckdb_encode(conn)
duckdb_encode_df = conn.table("financial_trx_encoded").to_df()
scikit_encode_df = scikit_encode(raw_data_df)

number_of_records_matching = (
    conn.from_df(duckdb_encode_df)
    .select("""
    transaction_id,
    ACH_onehot,
    UPI_onehot,
    card_onehot,
    wire_transfer_onehot,
    atm_onehot,
    mobile_onehot,
    pos_onehot,
    web_onehot,
    entertainment_onehot,
    grocery_onehot,
    online_onehot,
    other_onehot,
    restaurant_onehot,
    retail_onehot,
    travel_onehot,
    utilities_onehot,
    ordinal__transaction_type
 """)
    .union(
        conn.from_df(scikit_encode_df).select("""
        transaction_id,
        onehot__payment_channel_ACH,
        onehot__payment_channel_UPI,
        onehot__payment_channel_card,
        onehot__payment_channel_wire_transfer,
        onehot__device_used_atm,
        onehot__device_used_mobile,
        onehot__device_used_pos,
        onehot__device_used_web,
        onehot__merchant_category_entertainment,
        onehot__merchant_category_grocery,
        onehot__merchant_category_online,
        onehot__merchant_category_other,
        onehot__merchant_category_restaurant,
        onehot__merchant_category_retail,
        onehot__merchant_category_travel,
        onehot__merchant_category_utilities,
        ordinal__transaction_type
    """)
    )
    .distinct()
    .count("*")
    .fetchone()[0]
)

print(
    f"{number_of_records_matching} records have the same encoding in duckdb and scikit learn"
)

number_of_records_non_matching = (
    conn.from_df(duckdb_encode_df)
    .select("""
    transaction_id,
    ACH_onehot,
    UPI_onehot,
    card_onehot,
    wire_transfer_onehot,
    atm_onehot,
    mobile_onehot,
    pos_onehot,
    web_onehot,
    entertainment_onehot,
    grocery_onehot,
    online_onehot,
    other_onehot,
    restaurant_onehot,
    retail_onehot,
    travel_onehot,
    utilities_onehot,
    ordinal__transaction_type
 """)
    .except_(
        conn.from_df(scikit_encode_df).select("""
        transaction_id,
        onehot__payment_channel_ACH,
        onehot__payment_channel_UPI,
        onehot__payment_channel_card,
        onehot__payment_channel_wire_transfer,
        onehot__device_used_atm,
        onehot__device_used_mobile,
        onehot__device_used_pos,
        onehot__device_used_web,
        onehot__merchant_category_entertainment,
        onehot__merchant_category_grocery,
        onehot__merchant_category_online,
        onehot__merchant_category_other,
        onehot__merchant_category_restaurant,
        onehot__merchant_category_retail,
        onehot__merchant_category_travel,
        onehot__merchant_category_utilities,
        ordinal__transaction_type
    """)
    )
    .count("*")
    .fetchone()[0]
)

print(
    f"{number_of_records_non_matching} records have different encoding in duckdb and scikit learn"
)

x_train, x_test = scikit_split_data(scikit_encode_df)

conn.from_df(x_train).select("""
        * EXCLUDE ('time_since_last_transaction'),
        coalesce(time_since_last_transaction, avg(time_since_last_transaction) over ()) as time_since_last_transaction
""").to_table("financial_trx_training")

conn.from_df(x_test).set_alias("src").join(
    conn.sql(
        "from scaling_params('financial_trx_training', ['time_since_last_transaction'])"
    ),
    condition="1=1",
).select("""
    src.* EXCLUDE ('time_since_last_transaction'),
    coalesce(time_since_last_transaction, avg_time_since_last_transaction) as time_since_last_transaction
""").to_table("financial_trx_testing")

print("Reconcile feature scaling in training data")
duckdb_training_data_df = duckdb_get_transformed_training_data(conn)
scaling_steps_proc, scikit_training_data_df = scikit_get_transformed_training_data(
    x_train
)

# rounding to 8 decimals due to computation differences
number_of_records_matching = (
    conn.from_df(duckdb_training_data_df)
    .select("""
    transaction_id,
    round(ss_velocity_score, 8),
    round(min_max_spending_deviation_score, 8),
    round(min_max_time_since_last_transaction, 8),
    round(rs_amount, 8)
 """)
    .union(
        conn.from_df(scikit_training_data_df).select("""
        transaction_id,
        round(ss__velocity_score, 8),
        round(minmax__spending_deviation_score, 8),
        round(minmax_time_since_last_transaction__time_since_last_transaction, 8),
        round(rs__amount, 8)
    """)
    )
    .distinct()
    .count("*")
    .fetchone()[0]
)

print(
    f"{number_of_records_matching} records have the same scalers in duckdb and scikit learn training data"
)

number_of_records_non_matching = (
    conn.from_df(duckdb_training_data_df)
    .select("""
    transaction_id,
    round(ss_velocity_score, 8),
    round(min_max_spending_deviation_score, 8),
    round(min_max_time_since_last_transaction, 8),
    round(rs_amount, 8)
 """)
    .except_(
        conn.from_df(scikit_training_data_df).select("""
        transaction_id,
        round(ss__velocity_score, 8),
        round(minmax__spending_deviation_score, 8),
        round(minmax_time_since_last_transaction__time_since_last_transaction, 8),
        round(rs__amount, 8)
    """)
    )
    .count("*")
    .fetchone()[0]
)

print(
    f"{number_of_records_non_matching} records have different scalers in duckdb and scikit learn training data"
)

print("Reconcile feature scaling in training data")
duckdb_testing_data_df = duckdb_get_transformed_testing_data(conn)
scikit_testing_data_df = scikit_get_transformed_testing_data(scaling_steps_proc, x_test)

number_of_records_matching = (
    conn.from_df(duckdb_testing_data_df)
    .select("""
    transaction_id,
    round(ss_velocity_score, 8),
    round(min_max_spending_deviation_score, 8),
    round(min_max_time_since_last_transaction, 8),
    round(rs_amount, 8)
 """)
    .union(
        conn.from_df(scikit_testing_data_df).select("""
        transaction_id,
        round(ss__velocity_score, 8),
        round(minmax__spending_deviation_score, 8),
        round(minmax_time_since_last_transaction__time_since_last_transaction, 8),
        round(rs__amount, 8)
    """)
    )
    .distinct()
    .count("*")
    .fetchone()[0]
)

print(
    f"{number_of_records_matching} records have the same scalers in duckdb and scikit learn testing data"
)

number_of_records_non_matching = (
    conn.from_df(duckdb_testing_data_df)
    .select("""
    transaction_id,
    round(ss_velocity_score, 8),
    round(min_max_spending_deviation_score, 8),
    round(min_max_time_since_last_transaction, 8),
    round(rs_amount, 8)
 """)
    .except_(
        conn.from_df(scikit_testing_data_df).select("""
        transaction_id,
        round(ss__velocity_score, 8),
        round(minmax__spending_deviation_score, 8),
        round(minmax_time_since_last_transaction__time_since_last_transaction, 8),
        round(rs__amount, 8)
    """)
    )
    .count("*")
    .fetchone()[0]
)

print(
    f"{number_of_records_non_matching} records have different scalers in duckdb and scikit learn testing data"
)
