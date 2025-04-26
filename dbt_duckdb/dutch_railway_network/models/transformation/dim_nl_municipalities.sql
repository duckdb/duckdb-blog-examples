{{ config(materialized='table') }}

WITH covered_by_selection AS (
  SELECT
    id       AS municipality_id,
    statnaam AS municipality_name,
    geom     AS municipality_geometry,
    dim_prov.province_sk
  FROM st_read({{ source("geojson_external", "nl_municipalities") }}) AS dim_mun
  INNER JOIN {{ ref ("dim_nl_provinces") }} AS dim_prov
    ON st_covers(dim_prov.province_geometry, dim_mun.geom)
),
ordered_by_difference_area AS (
  SELECT
    id       AS municipality_id,
    statnaam AS municipality_name,
    geom     AS municipality_geometry,
    dim_prov.province_sk
  FROM st_read({{ source("geojson_external", "nl_municipalities") }}) AS dim_mun,
    {{ ref ("dim_nl_provinces") }} AS dim_prov
  WHERE NOT EXISTS (
      SELECT 1 FROM covered_by_selection
      WHERE dim_mun.id = covered_by_selection.municipality_id
    )
  QUALIFY row_number() OVER (
      PARTITION BY municipality_id
      ORDER BY st_area(st_difference(dim_mun.geom, province_geometry))
    ) = 1
)
SELECT
  {{ dbt_utils.generate_surrogate_key(['municipality_id']) }} AS municipality_sk,
  src.*,
  {{ common_columns() }}
FROM covered_by_selection AS src
UNION
SELECT
  {{ dbt_utils.generate_surrogate_key(['municipality_id']) }} AS municipality_sk,
  src.*,
  {{ common_columns() }}
FROM ordered_by_difference_area AS src
UNION
SELECT
  'unknown' AS municipality_sk,
  -1        AS municipality_id,
  'unknown' AS municipality_name,
  NULL      AS municipality_geometry,
  'unknown' AS province_sk,
  {{ common_columns() }}