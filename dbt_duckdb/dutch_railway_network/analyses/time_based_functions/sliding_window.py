import logging

from funcy import log_durations

from utils import get_duckdb_conn


@log_durations(logging.info)
def get_sliding_window(duckdb_conn):
    return duckdb_conn.sql(
        f"""
    SELECT
        station_service_time - INTERVAL '15' MINUTE AS window_start,
        station_service_time AS window_end,
        count(service_sk) OVER (
            ORDER BY station_service_time
                RANGE
                    BETWEEN INTERVAL '15' MINUTE PRECEDING
                    AND CURRENT ROW
        ) AS number_of_services
    FROM ams_traffic_v
    ORDER BY 3 DESC, 1
    LIMIT 5
    """
    )


@log_durations(logging.info)
def main():
    duckdb_conn = get_duckdb_conn()
    sliding_window = get_sliding_window(duckdb_conn)

    logging.info(sliding_window.show())


if __name__ == "__main__":
    main()
