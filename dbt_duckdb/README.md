Pre-requisites: docker, make, Python >= 3.12.

## Local execution

1. Startup PostgreSQL with `make setup-postgres`
2. Create Python virtual env with requirements: `make setup-python`
3. Build dbt models: `make run-dbt`
4. Serve chart: `make serve-chart`, the chart is served at: http://localhost:8888/charts.html

## Cleanup

1. Run `make clean-up`

## Misc

- Format SQL with `sqlfluff fix models`
- Generate ERD `dbt docs generate && dbterd run -t mermaid -s schema:main_public`
- Connect from DuckDB to PostgreSQL
    ```sql
    CREATE secret pg(
        type postgres,
        host '127.0.0.1',
        port '5466',
        database 'postgres',
        user 'postgres',
        password 'mysecretpassword'
    );
    ATTACH '' AS postgres_db (type postgres, schema 'main_public', secret pg);
    ```
- Generate schema files 
  ```bash 
  dbt run-operation generate_model_yaml --args '{"model_names": [], "upstream_descriptions":true}'
  ```
  
- If there is an issue with spatial, make sure to force [update the version](https://github.com/duckdb/duckdb-spatial/issues/508).

