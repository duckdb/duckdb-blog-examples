import logging
from pathlib import Path

import duckdb
from funcy import log_durations

logging.basicConfig(level=logging.INFO)

DB_DIR_PATH = f"{Path(__file__).parent.parent.parent.absolute()}/data"

PLOTS_DIR_PATH = f"{Path(__file__).parent.parent.absolute()}"


@log_durations(logging.info)
def get_duckdb_conn():
    duckdb_conn = duckdb.connect(
        f"{DB_DIR_PATH}/dutch_railway_network.duckdb",
        read_only=True,
    )
    duckdb_conn.sql("use main_main")
    return duckdb_conn
