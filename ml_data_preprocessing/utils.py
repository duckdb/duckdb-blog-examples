import csv
import time
from datetime import datetime

import duckdb


def timeit(method):
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        method_source = method.__name__.split("_")[0]
        execution_time = te - ts
        with open("execution_log.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now().isoformat(),
                    method_source,
                    method.__name__,
                    execution_time,
                ]
            )
        print(f"{method.__name__} took {(te - ts):.2f} seconds")
        return result

    return timed


def get_data_as_pd():
    conn = duckdb.connect()
    return conn.read_csv(
        "https://blobs.duckdb.org/data/financial_fraud_detection_dataset.csv"
    ).to_df()
