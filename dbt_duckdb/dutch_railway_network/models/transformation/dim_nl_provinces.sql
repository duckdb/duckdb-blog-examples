{{ config(materialized='table') }}

SELECT
    {{ dbt_utils.generate_surrogate_key(['id']) }} AS province_sk,
    id                               AS province_id,
    statnaam                         AS province_name,
    geom                             AS province_geometry,
    {{ common_columns() }}
FROM st_read({{ source("geojson_external", "nl_provinces") }}) AS src
UNION ALL
SELECT
    'unknown' AS province_sk,
    -1        AS province_id,
    'unknown' AS province_name,
    NULL      AS province_geometry,
    {{ common_columns() }}