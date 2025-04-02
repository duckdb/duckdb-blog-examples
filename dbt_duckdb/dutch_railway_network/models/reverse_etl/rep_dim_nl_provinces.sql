{{ config(materialized='table') }}

SELECT
  province_sk,
  province_name,
  ST_ASHEXWKB(province_geometry) AS province_geometry,
  {{ common_columns() }}
FROM {{ ref("dim_nl_provinces") }}