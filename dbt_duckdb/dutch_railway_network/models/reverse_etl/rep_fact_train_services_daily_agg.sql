{{
    config(
        materialized='incremental',
        incremental_strategy='delete+insert',
        unique_key='service_date, service_type, service_company, station_sk'
    )
}}

SELECT
    service_date,
    service_type,
    service_company,
    srv.station_sk,
    mn.municipality_sk,
    province_sk,
    count(*) AS number_of_rides,
    {{ common_columns() }}
FROM {{ ref ("fact_services") }} AS srv
INNER JOIN {{ ref("rep_dim_nl_train_stations") }} AS tr_st
    ON srv.station_sk = tr_st.station_sk
INNER JOIN {{ ref("rep_dim_nl_municipalities") }} AS mn
    ON tr_st.municipality_sk = mn.municipality_sk
WHERE service_arrival_cancelled IS FALSE

    {% if is_incremental() %}
        AND srv.invocation_id = (
            SELECT invocation_id FROM {{ ref("fact_services") }}
            ORDER BY last_updated_dt DESC LIMIT 1
        )
    {% endif %}
GROUP BY ALL