.PHONY: start-app clean-up

start-app:
	python -m venv venv_duckdb_streamlit && \
	source ./venv_duckdb_streamlit/bin/activate && \
	pip install -r requirements.txt && \
	streamlit run app.py

clean-up:
	rm -rf venv_duckdb_streamlit
