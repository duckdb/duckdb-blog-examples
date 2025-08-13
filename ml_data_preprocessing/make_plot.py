from pathlib import Path

import duckdb
import plotly.express as px

conn = duckdb.connect()

directory_path = Path(__file__).resolve().parent

execution_log_df = (
    conn.read_csv(f"{directory_path}/execution_log.csv")
    .select("""
    column1 as library,
    replace(column2, column1||'_', '') as step,
    round(column3, 2) as execution_time,
    column0,
    row_number() over (partition by column2 order by column0) as iteration_id
""")
    .filter("library in ('duckdb', 'scikit')")
    .filter(
        "step in ('encode', 'split_data', 'feature_scaling_training_data', 'feature_scaling_testing_data')"
    )
    .filter("iteration_id <= 1")
    .order("iteration_id, column0")
    .to_df()
)

px.bar(
    execution_log_df,
    x="step",
    y="execution_time",
    color="library",
    barmode="group",
    # facet_col="iteration_id",
    labels={
        "iteration_id": "Iteration",
        "step": "Step",
        "execution_time": "Execution Time in Seconds",
    },
    title="Data Preprocessing Benchmark",
).write_html(f"{directory_path}/benchmark.html")
