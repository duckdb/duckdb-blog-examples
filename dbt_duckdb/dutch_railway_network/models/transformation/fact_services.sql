{{ config(materialized='table') }}

SELECT
    {{ dbt_utils.generate_surrogate_key(['"Service:RDT-ID"', 'station_sk']) }}                                          AS service_sk,
    "Service:Date"                                                            AS service_date,
    "Service:Type"                                                            AS service_type,
    "Service:Company"                                                         AS service_company,
    station_sk,
    "Stop:Arrival time"                                                       AS station_arrival_time,
    "Stop:Departure time"                                                     AS station_departure_time,
    if("Stop:Arrival cancelled" IS NULL, FALSE, "Stop:Arrival cancelled")     AS service_arrival_cancelled,
    "Service:Train number"                                                    AS service_train_number,
    if("Stop:Departure cancelled" IS NULL, FALSE, "Stop:Departure cancelled") AS service_departure_cancelled,
    {{ common_columns() }}
FROM {{ source("external_db", "services") }} AS srv
INNER JOIN {{ ref("dim_nl_train_stations") }} AS tr_st
    ON srv."Stop:Station Code" = tr_st.station_code