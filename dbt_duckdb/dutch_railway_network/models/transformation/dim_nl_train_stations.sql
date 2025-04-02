{{ config(materialized='table') }}

SELECT
  {{ dbt_utils.generate_surrogate_key(['tr_st.code']) }}  AS station_sk,
  tr_st.id                                     AS station_id,
  tr_st.code                                   AS station_code,
  tr_st.name_long                              AS station_name,
  tr_st.type                                   AS station_type,
  ST_POINT(tr_st.geo_lng, tr_st.geo_lat)       AS station_geo_location,
  COALESCE(dim_mun.municipality_sk, 'unknown') municipality_sk,
  {{ common_columns() }}
FROM {{ source("external_db", "stations") }} AS tr_st
LEFT JOIN {{ ref ("dim_nl_municipalities") }} AS dim_mun
  ON ST_CONTAINS(
          dim_mun.municipality_geometry,
          ST_POINT(tr_st.geo_lng, tr_st.geo_lat)
  )
WHERE tr_st.country = 'NL'
