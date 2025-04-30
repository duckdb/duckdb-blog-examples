{{ config(materialized='view') }}

SELECT
    service_sk,
    if(station_arrival_time IS NULL, station_departure_time, station_arrival_time) AS station_service_time
FROM {{ ref("fact_services") }} AS srv
INNER JOIN {{ ref("dim_nl_train_stations") }} AS st
    ON srv.station_sk = st.station_sk
WHERE station_name = 'Amsterdam Centraal'
    AND (service_arrival_cancelled = FALSE OR service_departure_cancelled = FALSE)
