-- check that the province which contains the point is the same as the province associated through municipality
SELECT
  station_sk,
  province_sk
FROM {{ ref("dim_nl_train_stations") }} AS ts
INNER JOIN {{ ref("dim_nl_municipalities") }} AS mn
  ON ts.municipality_sk = mn.municipality_sk
WHERE ts.station_code NOT IN ('HLGH', 'EEM') -- issue with geo match
EXCEPT
SELECT
  station_sk,
  province_sk
FROM {{ ref("dim_nl_train_stations") }} AS ts
INNER JOIN {{ ref("dim_nl_provinces") }} AS p
  ON ST_CONTAINS(p.province_geometry, station_geo_location)
