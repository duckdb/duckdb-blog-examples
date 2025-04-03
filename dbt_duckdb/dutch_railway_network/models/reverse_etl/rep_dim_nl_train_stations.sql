{{ config(materialized='table') }}

SELECT
  station_sk,
  station_code,
  station_name,
  station_type,
  st_ashexwkb(station_geo_location) AS station_geo_location,
  municipality_sk,
  {{ common_columns() }}
FROM {{ ref("dim_nl_train_stations") }}