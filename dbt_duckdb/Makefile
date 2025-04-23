.PHONY: setup-postgres, force-clean-pg-container, setup-env, run-dbt, serve-chart, clean-up

setup-postgres:
	docker run --name postgres-dbt-duckdb -e POSTGRES_PASSWORD=mysecretpassword -p 5466:5432 -d postgres:17.4
	docker exec -it postgres-dbt-duckdb apt-get update
	docker exec -it postgres-dbt-duckdb apt-get install -y postgresql-17-postgis-3 postgis
	docker exec -it postgres-dbt-duckdb psql -U postgres -c 'create extension postgis'

setup-python:
	python -m venv venv_dbt_duckdb && \
	source venv_dbt_duckdb/bin/activate && \
	pip install -r requirements.txt

run-dbt:
	export DBT_DUCKDB_PG_PWD=mysecretpassword && \
	source venv_dbt_duckdb/bin/activate && \
	cd dutch_railway_network && \
	dbt deps && \
	dbt build

serve-chart:
	source venv_dbt_duckdb/bin/activate && \
	cd dutch_railway_network && \
	python analyses/build_charts.py && \
	python analyses/time_based_functions/tumbling_window.py && \
	python analyses/time_based_functions/sliding_window.py  && \
	python analyses/time_based_functions/session_window.py && \
	echo 'go to http://localhost:8888' && \
	python -m http.server 8888 -d analyses


force-clean-pg-container:
	docker rm -f postgres-dbt-duckdb

clean-up: force-clean-pg-container
	rm -rf venv_dbt_duckdb
