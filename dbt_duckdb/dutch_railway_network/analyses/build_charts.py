import json
from pathlib import Path

import plotly.express as px
import duckdb

dir_path = Path(__file__).parent.parent.absolute()

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="{asctime} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S.%s",
)


def main():
    with duckdb.connect() as con:
        con.sql("load spatial;")

        logging.info("Reading parquet")
        services = (
            con.read_parquet(
                f"{dir_path}/data/exports/nl_train_services_aggregate/*/*/*.parquet",
                hive_partitioning=True,
            )
            .filter("service_year=2024")
            .filter("province_sk != 'unknown'")
        )

        logging.info("Getting number of rides")

        province_summary_df = services.aggregate(
            """
            province_sk,
            province_name,
            sum(number_of_rides) as number_of_rides
        """
        ).df()

        with open(f"{dir_path}/data/exports/provinces.json", "r") as f:
            province_geojson = json.load(f)

        logging.info("Generating map")

        fig = px.choropleth_map(
            province_summary_df,
            geojson=province_geojson,
            locations="province_sk",
            featureidkey="properties.province_sk",
            color="number_of_rides",
            color_continuous_scale="peach",
            center=dict(lat=52.20528, lon=5.5),
            zoom=6.5,
            height=800,
            width=800,
            title="Train Rides, Dutch Provinces, 2024",
            labels={"number_of_rides": "Number of Rides"},
            template="plotly_dark",
            hover_name="province_name",
        )

        logging.info("Saving map")
        fig.write_html(f"{dir_path}/analyses/charts.html")

        logging.info("Done")


if __name__ == "__main__":
    main()
