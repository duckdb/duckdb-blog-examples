{{ config(
    materialized='table',
    post_hook = """
    call postgres_execute(
        '{{ this.database }}',
        '
            alter table {{ this.schema }}.rep_dim_nl_train_stations
            alter column station_geo_location type geometry
            using  ST_GeomFromWKB(decode(station_geo_location, ''hex''))
        '
    )
    """) }}

SELECT
    station_sk,
    station_code,
    station_name,
    station_type,
    st_ashexwkb(station_geo_location) AS station_geo_location,
    municipality_sk,
    {{ common_columns() }}
FROM {{ ref("dim_nl_train_stations") }}