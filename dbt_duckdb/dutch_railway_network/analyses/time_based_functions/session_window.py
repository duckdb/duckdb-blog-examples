import logging

import pandas as pd
import plotly.express as px
from funcy import log_durations

from utils import PLOTS_DIR_PATH, get_duckdb_conn


@log_durations(logging.info)
def get_session_window(duckdb_conn):
    return duckdb_conn.sql(
        """
    WITH ams_daily_traffic AS (
        SELECT
            service_sk,
            station_service_time,
            lag(station_service_time) OVER (
                PARTITION BY station_service_time::DATE
                ORDER BY station_service_time
            ) AS previous_service_time,
            date_diff('minute', previous_service_time, station_service_time) AS gap_minutes
        FROM ams_traffic_v
        WHERE hour(station_service_time) BETWEEN 6 AND 23
    ), window_calculation AS (
            SELECT
                service_sk,
                station_service_time,
                station_service_time::DATE AS station_service_date,
                gap_minutes,
                IF(gap_minutes >= 10 OR gap_minutes IS NULL, 1, 0) new_session,
                sum(new_session) OVER (
                    PARTITION BY station_service_date
                    ORDER BY station_service_time ROWS UNBOUNDED PRECEDING
                ) AS session_id_in_day
           FROM ams_daily_traffic
    ), session_window AS (
        SELECT
            station_service_date,
            session_id_in_day,
            max(gap_minutes)          AS gap_minutes,
            min(station_service_time) AS window_start,
            max(station_service_time) AS window_end,
            count(service_sk)         AS number_of_arrivals
        FROM window_calculation
        GROUP BY ALL
    )
        SELECT
            station_service_date,
            session_id_in_day,
            max(gap_minutes)          AS gap_minutes,
            min(station_service_time) AS window_start,
            max(station_service_time) AS window_end,
            count(service_sk)         AS number_of_services
        FROM window_calculation
        GROUP BY ALL
    """
    )


@log_durations(logging.info)
def get_top5_day_with_most_gaps(duckdb_conn):
    session_window = get_session_window(duckdb_conn)

    return (
        session_window.aggregate(
            """
        station_service_date,
        max(ceil(date_diff('minute', window_start, window_end) / 60)) AS number_of_hours_without_gap,
        count(*) AS number_of_sessions
        """
        )
        .filter("number_of_hours_without_gap")
        .order("number_of_sessions desc, station_service_date")
        .limit(5)
    )


@log_durations(logging.info)
def save_session_window_px(duckdb_conn, day_with_most_gaps):
    df = (
        get_session_window(duckdb_conn)
        .filter(f"station_service_date = '{day_with_most_gaps}'")
        .df()
    )
    unique_gap = df["gap_minutes"].sort_values().unique()
    fig = px.timeline(
        df,
        x_start="window_start",
        x_end="window_end",
        y="gap_minutes",
        title=f"Session Windows on {day_with_most_gaps}",
        category_orders={"gap_minutes": unique_gap},
    )

    fig.update_yaxes(autorange=True)

    all_ticks = pd.concat([df["window_start"], df["window_end"]]).sort_values().unique()
    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=all_ticks,
            tickformat="%H:%M",
            tickangle=90,
            tickfont=dict(size=8, family="Arial Bold"),
        ),
        xaxis_range=[
            df["window_start"].min() - pd.Timedelta(minutes=5),
            df["window_end"].max() + pd.Timedelta(minutes=5),
        ],
        xaxis_title="Time",
        yaxis_title="Duration of Service Inactivity, in minutes",
    )

    for t in all_ticks:
        fig.add_vline(x=t, line_width=1, line_dash="dot", line_color="gray")

    fig.update_layout(yaxis={"tickvals": unique_gap, "type": "category"})

    fig.update_yaxes(categoryorder="array", categoryarray=[str(v) for v in unique_gap])

    fig.write_html(f"{PLOTS_DIR_PATH}/session_window.html")


@log_durations(logging.info)
def main():
    duckdb_conn = get_duckdb_conn()
    most_detected_gaps = get_top5_day_with_most_gaps(duckdb_conn)
    logging.info(most_detected_gaps.show())
    save_session_window_px(
        duckdb_conn, day_with_most_gaps=most_detected_gaps.fetchone()[0]
    )


if __name__ == "__main__":
    main()
