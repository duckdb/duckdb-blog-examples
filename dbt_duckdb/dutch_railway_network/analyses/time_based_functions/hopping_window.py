import logging

from funcy import log_durations

from utils import get_duckdb_conn


@log_durations(logging.info)
def get_hopping_window(duckdb_conn):
    return duckdb_conn.sql(
        """
    WITH time_range AS (
        SELECT
            range AS window_start,
            window_start + INTERVAL '15' MINUTE AS window_end -- window size of 15 minutes
        FROM range(
            '2024-01-01 00:00:00'::TIMESTAMP,
            '2025-01-01 00:00:00'::TIMESTAMP,
            INTERVAL '5' MINUTE -- hopping size of 5 minute
        )
    )
    SELECT
        window_start,
        window_end,
        count(service_sk) AS number_of_services
    FROM ams_traffic_v
    INNER JOIN time_range AS ts
        ON station_service_time >= ts.window_start AND station_service_time < ts.window_end
    GROUP BY ALL
    ORDER BY 3 DESC, 1 ASC
    LIMIT 5
    """
    )


@log_durations(logging.info)
def main():
    duckdb_conn = get_duckdb_conn()
    hopping_window = get_hopping_window(duckdb_conn)

    logging.info(hopping_window.show())


if __name__ == "__main__":
    main()
