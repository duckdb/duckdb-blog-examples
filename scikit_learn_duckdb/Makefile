.PHONY: setup-python run-marimo

setup-python:
	python -m venv venv_scikit_learn_duckdb && \
	source venv_scikit_learn_duckdb/bin/activate && \
	pip install -r requirements.txt

run-marimo:
	source venv_scikit_learn_duckdb/bin/activate && \
	marimo edit predict_penguin_species.py