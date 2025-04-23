import logging

import plotly.express as px
from funcy import log_durations

from utils import PLOTS_DIR_PATH, get_duckdb_conn


@log_durations(logging.info)
def get_hour_tumbling_window_df(duckdb_conn):
    return duckdb_conn.sql(
        """
    SELECT
        date_trunc('hour', station_service_time) station_service_time_hour,
        count(*) AS number_of_services
    FROM ams_traffic_v
    WHERE year(station_service_time) = 2024
    GROUP BY ALL
    ORDER BY 1
    """
    ).df()


@log_durations(logging.info)
def save_hour_tumbling_px(duckdb_conn):
    fig = px.line(
        get_hour_tumbling_window_df(duckdb_conn),
        x="station_service_time_hour",
        y="number_of_services",
        title="Hourly Train Services, 2024",
    )

    fig.write_html(f"{PLOTS_DIR_PATH}/hour_tumbling_window.html")


@log_durations(logging.info)
def get_quarter_tumbling_window_df(duckdb_conn):
    return duckdb_conn.sql(
        """
    SELECT
        strftime('%H:%M', time_bucket(
            INTERVAL '15' MINUTE, -- bucket width
            station_service_time,
            INTERVAL '0' MINUTE -- offset
        ))  AS station_service_time_hour_quarter,
        count(*) / 366 AS number_of_services
    FROM ams_traffic_v
    WHERE year(station_service_time) = 2024
    GROUP BY ALL
    ORDER BY 1
    """
    ).df()


@log_durations(logging.info)
def save_quarter_tumbling_px(duckdb_conn):
    fig = px.line(
        get_quarter_tumbling_window_df(duckdb_conn),
        x="station_service_time_hour_quarter",
        y="number_of_services",
        title="Average Number of Train Services, per 15 minutes, 2024",
    )

    fig.update_layout(xaxis={"dtick": 1})
    fig.write_html(f"{PLOTS_DIR_PATH}/hour_quarter_tumbling_window.html")


@log_durations(logging.info)
def main():
    duckdb_conn = get_duckdb_conn()
    save_hour_tumbling_px(duckdb_conn)
    save_quarter_tumbling_px(duckdb_conn)


if __name__ == "__main__":
    main()
