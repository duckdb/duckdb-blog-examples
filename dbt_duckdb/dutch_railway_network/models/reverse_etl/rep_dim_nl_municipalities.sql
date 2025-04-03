{{ config(materialized='table') }}

SELECT
  municipality_sk,
  municipality_name,
  st_ashexwkb(municipality_geometry) AS municipality_geometry,
  province_sk,
  {{ common_columns() }}
FROM {{ ref("dim_nl_municipalities") }}