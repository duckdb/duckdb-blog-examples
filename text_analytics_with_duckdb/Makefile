.PHONY: setup-python, run-marimo

setup-python:
	python -m venv venv_text_analytics_with_duckdb && \
	source venv_text_analytics_with_duckdb/bin/activate && \
	pip install -r requirements.txt

run-marimo:
	source venv_text_analytics_with_duckdb/bin/activate && \
	TOKENIZERS_PARALLELISM=false marimo edit text_analytics_with_duckdb.py
