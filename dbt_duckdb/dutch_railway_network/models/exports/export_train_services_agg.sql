{{ config(
    materialized='external',
    location="data/exports/nl_train_services_aggregate",
    options={"partition_by": "service_year, service_month", "overwrite": True}
    )
}}

SELECT
  YEAR(service_date)  AS service_year,
  MONTH(service_date) AS service_month,
  service_type,
  service_company,
  tr_st.station_sk,
  tr_st.station_name,
  m.municipality_sk,
  m.municipality_name,
  p.province_sk,
  p.province_name,
  COUNT(*)            AS number_of_rides
FROM {{ ref ("fact_services") }} AS srv
INNER JOIN {{ ref("dim_nl_train_stations") }} AS tr_st
  ON srv.station_sk = tr_st.station_sk
INNER JOIN {{ ref("dim_nl_municipalities") }} AS m
  ON tr_st.municipality_sk = m.municipality_sk
INNER JOIN {{ ref("dim_nl_provinces") }} AS p
  ON m.province_sk = p.province_sk
WHERE service_year = {{ var('execution_year') }}
GROUP BY ALL