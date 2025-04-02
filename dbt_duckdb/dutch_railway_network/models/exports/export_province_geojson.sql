 {{ config(
    materialized='external',
    location="data/exports/provinces.json"
    )
}}

WITH province_agg AS (
  SELECT
    JSON_GROUP_ARRAY(
      JSON_OBJECT(
        'type', 'Feature',
        'properties', JSON_OBJECT('province_sk', province_sk),
        'geometry', ST_ASGEOJSON(province_geometry)
      )
    ) AS features
  FROM {{ ref("dim_nl_provinces") }}
)
SELECT
  'FeatureCollection' AS type,
  features
FROM province_agg