{{ config(
    materialized='table',
    post_hook = """
    call postgres_execute(
        '{{ this.database }}',
        '
            alter table {{ this.schema }}.rep_dim_nl_provinces
            alter column province_geometry type geometry
            using  ST_GeomFromWKB(decode(province_geometry, ''hex''))
        '
    )
    """)
}}

SELECT
    province_sk,
    province_name,
    st_ashexwkb(province_geometry) AS province_geometry,
    {{ common_columns() }}
FROM {{ ref("dim_nl_provinces") }}