 {{ config(
    materialized='external',
    location="data/exports/provinces.json"
    )
}}

WITH province_agg AS (
  SELECT
    json_group_array(
      json_object(
        'type', 'Feature',
        'properties', json_object('province_sk', province_sk),
        'geometry', st_asgeojson(province_geometry)
      )
    ) AS features
  FROM {{ ref("dim_nl_provinces") }}
)
SELECT
  'FeatureCollection' AS type,
  features
FROM province_agg