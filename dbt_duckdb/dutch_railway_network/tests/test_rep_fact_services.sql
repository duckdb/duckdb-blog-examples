SELECT COUNT(*)
FROM {{ source("external_db", "services") }}
WHERE IF("Stop:Arrival cancelled" is null, false, "Stop:Arrival cancelled") IS FALSE
  AND "Stop:Station Code" IN (
    SELECT code
    FROM {{ source("external_db", "stations") }}
    WHERE country = 'NL'
  )
EXCEPT
SELECT SUM(number_of_rides)
FROM {{ ref('rep_fact_train_services_daily_agg') }}