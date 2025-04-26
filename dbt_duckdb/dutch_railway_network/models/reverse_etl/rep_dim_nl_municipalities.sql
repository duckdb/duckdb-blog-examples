{{ config(
    materialized='table',
    post_hook = """
    call postgres_execute(
        '{{ this.database }}',
        '
            alter table {{ this.schema }}.rep_dim_nl_municipalities
            alter column municipality_geometry type geometry
            using ST_GeomFromWKB(decode(municipality_geometry, ''hex''))
        '
    )
    """
    )
 }}

SELECT
  municipality_sk,
  municipality_name,
  st_ashexwkb(municipality_geometry) AS municipality_geometry,
  province_sk,
  {{ common_columns() }}
FROM {{ ref("dim_nl_municipalities") }}