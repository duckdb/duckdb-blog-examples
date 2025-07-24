.PHONY: setup-python reconcile-results benchmark-duckdb benchmark-scikit-learn

setup-python:
	python -m venv venv_ml_data_preprocessing && \
	source venv_ml_data_preprocessing/bin/activate && \
	pip install -r requirements.txt

reconcile-results:
	source venv_ml_data_preprocessing/bin/activate && \
	python reconcile_results.py

benchmark-duckdb:
	source venv_ml_data_preprocessing/bin/activate && \
	python benchmark_duckdb.py

benchmark-scikit-learn:
	source venv_ml_data_preprocessing/bin/activate && \
	python benchmark_scikit_learn.py

plot:
	source venv_ml_data_preprocessing/bin/activate && \
	python make_plot.py
