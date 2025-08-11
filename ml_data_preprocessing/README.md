# Data preprocessing with DuckDB

1. Run `make setup-python`, to create venv with required dependencies
2. Run `make reconcile-results`, to compare data preprocessing results with DuckDB versus scikit-learn
3. Run `make benchmark-duckdb`, to get the execution time for DuckDB
4. Run `make benchmark-scikit-learn`, to the execution time for scikit-learn

The download time of the data from the https location is not taken into account.
