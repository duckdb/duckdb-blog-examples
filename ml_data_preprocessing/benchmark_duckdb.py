import duckdb

from utils import get_data_as_pd, timeit
import duckdb

from utils import get_data_as_pd, timeit


@timeit
def duckdb_save_pd_as_table(duckdb_conn):
    raw_data_df = get_data_as_pd()
    duckdb_conn.from_df(raw_data_df).to_table("financial_trx")


@timeit
def duckdb_create_macros(duckdb_conn):
    duckdb_conn.sql("""
        CREATE OR REPLACE MACRO scaling_params(table_name, column_list) AS TABLE
        FROM query_table(table_name)
        SELECT
            avg(columns(column_list)) as 'avg_\\0',
            stddev_pop(columns(column_list)) as 'std_\\0',
            min(columns(column_list)) as 'min_\\0',
            max(columns(column_list)) as 'max_\\0',
            quantile_cont(columns(column_list), 0.25) AS 'q25_\\0',
            quantile_cont(columns(column_list), 0.50) AS 'q50_\\0',
            quantile_cont(columns(column_list), 0.75) AS 'q75_\\0',
            median(columns(column_list)) as 'median_\\0';    
    """)

    duckdb_conn.sql("""
    CREATE OR REPLACE MACRO standard_scaler(val, avg_val, std_val) AS
        (val - avg_val)/std_val;
    """)

    duckdb_conn.sql("""
    CREATE OR REPLACE MACRO min_max_scaler(val, min_val, max_val) AS
        (val - min_val)/nullif(max_val - min_val, 0);
    """)

    duckdb_conn.sql("""
    CREATE OR REPLACE MACRO robust_scaler(val, q25_val, q50_val, q75_val) AS
        (val - q50_val)/nullif(q75_val - q25_val, 0);
    """)


@timeit
def duckdb_encode(duckdb_conn):
    duckdb_conn.sql("drop table if exists financial_trx_encoded")
    duckdb_conn.sql("""
        WITH
            onehot_payment_channel AS (
                PIVOT financial_trx
                ON payment_channel
                USING COALESCE (MAX (payment_channel=payment_channel):: INT,0) AS 'onehot'
                GROUP BY payment_channel
            ),
            onehot_device_used AS (
                PIVOT financial_trx
                ON device_used
                USING COALESCE (MAX (device_used=device_used):: INT,0) AS 'onehot'
                GROUP BY device_used
            ),
            onehot_merchant_category AS (
                PIVOT financial_trx
                ON merchant_category
                USING COALESCE (MAX (merchant_category=merchant_category):: INT,0) AS 'onehot'
                GROUP BY merchant_category
            ),
            trx_type_ordinal_encoded AS (
                SELECT
                    transaction_type,
                    row_number() over (order by transaction_type) - 1 AS ordinal__transaction_type
              FROM (
                   SELECT DISTINCT transaction_type
                   FROM financial_trx
                   )
            )
            FROM financial_trx trx
        JOIN onehot_payment_channel USING (payment_channel)
        JOIN onehot_device_used USING (device_used)
        JOIN onehot_merchant_category USING (merchant_category)
        JOIN trx_type_ordinal_encoded USING (transaction_type)
    """).to_table("financial_trx_encoded")
    return duckdb_conn.table("financial_trx_encoded").to_df()


@timeit
def duckdb_split_data(duckdb_conn):
    duckdb_conn.sql("set threads=1")
    duckdb_conn.sql("drop table if exists financial_trx_training")
    duckdb_conn.sql("""
        SELECT
            *
        FROM financial_trx_encoded
        USING SAMPLE 80 PERCENT (reservoir, 256)
    """).to_table("financial_trx_training")

    duckdb_conn.sql("set threads=8")
    duckdb_conn.sql("drop table if exists financial_trx_testing")
    duckdb_conn.sql("""
        SELECT
            src.*
        FROM financial_trx_encoded src 
        WHERE NOT EXISTS (FROM financial_trx_training src_t WHERE src.transaction_id = src_t.transaction_id)
    """).to_table("financial_trx_testing")

    return duckdb_conn.table("financial_trx_training").to_df(), duckdb_conn.table("financial_trx_testing").to_df()


@timeit
def duckdb_feature_scaling_training_data(duckdb_conn):
    return duckdb_conn.sql("""
        SELECT
            transaction_id,
            * like '%onehot%',
            ordinal__transaction_type,
            ss_velocity_score: standard_scaler(
                velocity_score,
                avg_velocity_score,
                std_velocity_score
            ),
            min_max_spending_deviation_score: min_max_scaler(
                spending_deviation_score,
                min_spending_deviation_score,
                max_spending_deviation_score
            ),
            min_max_time_since_last_transaction : min_max_scaler(
                coalesce(time_since_last_transaction, avg_time_since_last_transaction),
                min_time_since_last_transaction,
                max_time_since_last_transaction
            ),
            rs_amount: robust_scaler(
                amount,
                q25_amount,
                q50_amount,
                q75_amount
            )
        FROM financial_trx_training,
             scaling_params(
                 'financial_trx_training',
                 ['velocity_score', 'spending_deviation_score', 'amount', 'time_since_last_transaction']
             )
    """).to_df()


@timeit
def duckdb_feature_scaling_testing_data(duckdb_conn):
    return duckdb_conn.sql("""
        SELECT
            transaction_id,
            * like '%onehot%',
            ordinal__transaction_type,
            ss_velocity_score: standard_scaler(
                velocity_score,
                avg_velocity_score,
                std_velocity_score
            ),
            min_max_spending_deviation_score: min_max_scaler(
                spending_deviation_score,
                min_spending_deviation_score,
                max_spending_deviation_score
            ),
            min_max_time_since_last_transaction : min_max_scaler(
                coalesce(time_since_last_transaction, avg_time_since_last_transaction),
                min_time_since_last_transaction,
                max_time_since_last_transaction
            ),
            rs_amount: robust_scaler(
                amount,
                q25_amount,
                q50_amount,
                q75_amount
            )
        FROM financial_trx_testing,
             scaling_params(
                 'financial_trx_training',
                 ['velocity_score', 'spending_deviation_score', 'amount', 'time_since_last_transaction']
             )
    """).to_df()


@timeit
def main():
    conn = duckdb.connect()

    duckdb_create_macros(conn)

    duckdb_save_pd_as_table(conn)

    for i in range(1, 2):
        print(f"Iteration {i}")
        encoded_df = duckdb_encode(conn)
        x_train_df, x_test_df = duckdb_split_data(conn)
        x_train_transformed_df = duckdb_feature_scaling_training_data(conn)
        x_test_transformed_df = duckdb_feature_scaling_testing_data(conn)


if __name__ == "__main__":
    main()
