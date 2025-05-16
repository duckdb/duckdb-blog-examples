import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import logging
    import pickle

    from datetime import datetime
    from decimal import Decimal

    import duckdb
    import numpy as np
    import orjson
    import plotly.express as px
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    return (
        Decimal,
        RandomForestClassifier,
        datetime,
        duckdb,
        np,
        orjson,
        pickle,
        px,
        train_test_split,
    )


@app.cell
def _(duckdb):
    # read the csv data from external location and exclude records with null values and alter column type
    def process_palmerpenguins_data(duckdb_conn):
        duckdb_conn.read_csv(
            "http://blobs.duckdb.org/data/penguins.csv"
        ).filter("columns(*)::text != 'NA'").filter("columns(*) is not null").select(
            "*, row_number() over () as observation_id"
        ).to_table(
            "penguins_data"
        )

        duckdb_conn.sql(
            "alter table penguins_data alter bill_length_mm  set data type decimal(5, 2)"
        )
        duckdb_conn.sql(
            "alter table penguins_data alter bill_depth_mm  set data type decimal(5, 2)"
        )
        duckdb_conn.sql("alter table penguins_data alter body_mass_g set data type integer")
        duckdb_conn.sql(
            "alter table penguins_data alter flipper_length_mm set data type integer"
        )

    duckdb_conn = duckdb.connect()

    process_palmerpenguins_data(duckdb_conn=duckdb_conn)

    return (duckdb_conn,)


@app.cell
def _(duckdb_conn, px):
    # plot species and island
    px.bar(
        duckdb_conn.table("penguins_data").aggregate(
            "species, island, count(*) as number_of_observations").order("island, species").df(),
        x="island",
        y="number_of_observations",
        color="species",
        title="Palmer Penguins Observations",
        barmode="group",
        labels={
            "number_of_observations": "Number of Observations",
            "island": "Island"
        }
    )
    return


@app.cell
def _(duckdb_conn, px):
    # plot features per species
    px.scatter(
        duckdb_conn.table("penguins_data").df(),
        x="bill_length_mm",
        y="bill_depth_mm",
        size="body_mass_g",
        color="species",
        title="Penguins Observations, bill length and depth, per species",
        labels={
            "bill_length_mm": "Bill Length in mm",
            "bill_depth_mm": "Bill Depth in mm"
        }
    )
    return


@app.cell
def _(duckdb_conn):
    # analyze the data
    duckdb_conn.table("penguins_data").describe().df()
    return


@app.cell
def _(duckdb_conn):
    # instead of label encoding, we create reference tables
    def process_reference_data(duckdb_conn):
        for feature in ["species", "island"]:
            duckdb_conn.sql(f"drop table if exists {feature}_ref")
            (
                duckdb_conn.table("penguins_data")
                .select(feature)
                .unique(feature)
                .row_number(
                    window_spec=f"over (order by {feature})", projected_columns=feature
                )
                .select(f"{feature}, #2 - 1 as {feature}_id")
                .to_table(f"{feature}_ref")
            )
            duckdb_conn.table(f"{feature}_ref").show()

    process_reference_data(duckdb_conn)

    return


@app.cell
def _(train_test_split):
    def train_split_data(selection_query):
        X_df = selection_query.select(
            "bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, island_id, observation_id, species_id"
        ).order("observation_id").df()
        y_df = [
            x[0] 
            for x in selection_query.order("observation_id").select("species_id").fetchall()
        ]

        num_test = 0.30
        return train_test_split(X_df, y_df, test_size=num_test)
    return (train_split_data,)


@app.cell
def _(RandomForestClassifier, pickle, train_split_data):
    def get_model(selection_query):
        X_train, X_test, y_train, y_test = train_split_data(selection_query)

        model = RandomForestClassifier(n_estimators=1, max_depth=2, random_state=5)

        model.fit(X_train.drop(["observation_id", "species_id"], axis=1).values, y_train)

        pickle.dump(model, open("./model/penguin_model.sav", "wb"))

        print(f" Accuracy score is: {model.score(
            X_test.drop(["observation_id", "species_id"], axis=1).values, y_test
        )}")
    return (get_model,)


@app.cell
def _(duckdb_conn, get_model, pickle):
    selection_query = (
        duckdb_conn.table("penguins_data")
        .join(duckdb_conn.table("island_ref"), condition="island")
        .join(duckdb_conn.table("species_ref"), condition="species")
    )

    get_model(selection_query)

    model = pickle.load(open("./model/penguin_model.sav", "rb"))
    return model, selection_query


@app.cell
def _(duckdb_conn, model, selection_query):
    # get predictions with pandas and duckdb in python

    predicted_df = selection_query.select(
        "bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, island_id, observation_id, species_id"
    ).df()

    predicted_df["predicted_species_id"] = model.predict(
        predicted_df.drop(["observation_id", "species_id"], axis=1).values
    )

    (
        duckdb_conn.table("predicted_df")
        .select("observation_id", "species_id", "predicted_species_id")
        .filter("species_id != predicted_species_id")
    )

    return (predicted_df,)


@app.cell
def _(duckdb_conn, mo, predicted_df):
    _df = mo.sql(
        f"""
        -- directly with SQL

        select observation_id, species_id, predicted_species_id
        from predicted_df
        where species_id != predicted_species_id
        """,
        engine=duckdb_conn
    )
    return


@app.cell
def _(Decimal, duckdb_conn, pickle, selection_query):
    # get predictions with duckdb udf, row by row

    def get_prediction_per_row(
        bill_length_mm: Decimal, bill_depth_mm: Decimal, flipper_length_mm: int, body_mass_g: int, island_id: int
    ) -> int:
        model = pickle.load(open("./model/penguin_model.sav", "rb"))
        return int(
            model.predict(
                [
                    [
                        bill_length_mm,
                        bill_depth_mm,
                        flipper_length_mm,
                        body_mass_g,
                        island_id,
                    ]
                ]
            )[0]
        )

    try:
        duckdb_conn.remove_function("predict_species_per_row")
    except Exception:
        pass
    finally:
        duckdb_conn.create_function(
                "predict_species_per_row", get_prediction_per_row, return_type=int
            )

    selection_query.select(
            """
            observation_id,
            species_id,
            predict_species_per_row(
                bill_length_mm, 
                bill_depth_mm, 
                flipper_length_mm, 
                body_mass_g, 
                island_id
            ) as predicted_species_id
        """
        ).filter("species_id != predicted_species_id")
    return


@app.cell
def _(Decimal, datetime, duckdb, duckdb_conn, np, orjson, pickle):
    # get predictions with duckdb udf, full / batch style

    def get_prediction_per_batch(input_data: dict[str, list[Decimal | int ]]) -> np.ndarray:
        """
        input_data example:
            {
                "bill_length_mm": [40.5],
                "bill_depth_mm": [41.5],
                "flipper_length_mm: [250],
                "body_mass_g": [3000],
                "island_id": [1]
            }
        """
        model = pickle.load(open("./model/penguin_model.sav", "rb"))

        st_dt = datetime.now()

        input_data_parsed = orjson.loads(input_data)

        print(f"JSON parsing took {(datetime.now() - st_dt).total_seconds()} seconds")

        st_dt = datetime.now()

        input_data_converted_to_numpy = np.stack(tuple(input_data_parsed.values()), axis=1)

        print(f"Converting to numpy took {(datetime.now() - st_dt).total_seconds()} seconds")

        return model.predict(input_data_converted_to_numpy)

    try:
        duckdb_conn.remove_function("predict_species_per_batch")
    except Exception:
        pass
    finally:
        duckdb_conn.create_function(
            "predict_species_per_batch",
            get_prediction_per_batch,
            return_type=duckdb.typing.DuckDBPyType(list[int]),
        )


    def get_selection_query_for_batch(selection_query):
        return (
            selection_query
            .aggregate("""
                json_object(
                    'bill_length_mm', array_agg(bill_length_mm),
                    'bill_depth_mm', array_agg(bill_depth_mm),
                    'flipper_length_mm', array_agg(flipper_length_mm),
                    'body_mass_g', array_agg(body_mass_g),
                    'island_id', array_agg(island_id)
                ) as input_data,
                struct_pack(
                    observation_id := array_agg(observation_id),
                    species_id := array_agg(species_id),
                    predicted_species_id := predict_species_per_batch(input_data)
                ) as output_data
            """)
            .select("""
                unnest(output_data.observation_id) as observation_id,
                unnest(output_data.species_id) as species_id,
                unnest(output_data.predicted_species_id) as predicted_species_id
            """)
        )

    return (get_selection_query_for_batch,)


@app.cell
def _(get_selection_query_for_batch, selection_query):
    # mass retrieval
    get_selection_query_for_batch(selection_query).filter("species_id != predicted_species_id").show()

    return


@app.cell
def _(get_selection_query_for_batch, selection_query):
    # batch style
    for i in range(4):
        (
            get_selection_query_for_batch(
                selection_query
                .order("observation_id")
                .limit(100, offset=100*i)
                .select("*")
            )
            .filter("species_id != predicted_species_id")
        ).show()
    return


@app.cell
def _(duckdb_conn, selection_query):
    def generate_dummy_data(duckdb_conn, selection_query):
        duckdb_conn.sql("drop table if exists dummy_generated_data")
        selection_query.filter("1 = 0").select(
            "bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, island_id, observation_id, species_id, species"
        ).to_table("dummy_generated_data")

        for idx, rec in enumerate(
            selection_query.aggregate("""
                island_id, 
                species_id, 
                min(bill_length_mm)::int as min_bill_length_mm, 
                max(bill_length_mm)::int as max_bill_length_mm, 
                min(bill_depth_mm)::int as min_bill_depth_mm, 
                max(bill_depth_mm)::int as max_bill_depth_mm, 
                min(flipper_length_mm) as min_flipper_length_mm,
                max(flipper_length_mm) as max_flipper_length_mm,
                min(body_mass_g) as min_body_mass_g,
                max(body_mass_g) as max_body_mass_g
            """).fetchall()
        ):
            bill_length_range = duckdb_conn.sql(f"from range({rec[2]}, {rec[3]})").select(
                "range as bill_length"
            )

            bill_depth_range = duckdb_conn.sql(f"from range({rec[4]}, {rec[5]})").select(
                "range as bill_depth"
            )

            flipper_length_range = duckdb_conn.sql(
                f"from range({rec[6]}, {rec[7]})"
            ).select("range as flipper_length")

            body_mass_range = duckdb_conn.sql(f"from range({rec[8]}, {rec[9]})").select(
                "range as body_mass"
            )

            dummy_range = duckdb_conn.sql("from range(1,10)").set_alias(
                "dummy_range"
            )

            sql_query = (
                dummy_range.join(bill_length_range, condition="1 = 1")
                .join(bill_depth_range, condition="1 = 1")
                .join(flipper_length_range, condition="1 = 1")
                .join(body_mass_range, condition="1 = 1")
                .join(duckdb_conn.table("species_ref"), condition=f"species_id = {rec[1]}")
                .select(
                    f"""
                     bill_length + 10 ** 1/range as bill_length_mm, 
                     bill_depth + 10 ** 1/range as bill_depth_mm, 
                     flipper_length + (10 ** 1/range)::int as flipper_length_mm, 
                     body_mass + (10 ** 1/range)::int as body_mass_g, 
                     {rec[0]} as island_id,
                     null as observation_id,
                     species_id,
                     species
                """
                )
            ).sql_query()

            duckdb_conn.sql(f"select * from ({sql_query}) using sample 30%").insert_into("dummy_generated_data")

    generate_dummy_data(duckdb_conn, selection_query)

    duckdb_conn.table("dummy_generated_data").count("*")

    return


@app.cell
def _(duckdb_conn):
    duckdb_conn.sql("select * from dummy_generated_data using sample 10%").to_table("sample_dummy_data")
    return


@app.cell
def _(duckdb_conn, get_selection_query_for_batch):
    (
        get_selection_query_for_batch(duckdb_conn.table("sample_dummy_data"))
        .aggregate("""
            sum(if(species_id = predicted_species_id, 1, 0)) number_of_correct_predictions,
            sum(if(species_id = predicted_species_id, 0, 1)) number_of_incorrect_predictions
        """)
    )
    return


@app.cell
def _(duckdb_conn, model):
    predicted_dummy_df = duckdb_conn.table("sample_dummy_data").select(
        "bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, island_id, observation_id, species_id"
    ).df()

    predicted_dummy_df["predicted_species_id"] = model.predict(
        predicted_dummy_df.drop(["observation_id", "species_id"], axis=1).values
    )

    (
        duckdb_conn.table("predicted_dummy_df")
        .select("observation_id", "species_id", "predicted_species_id")
        .aggregate("""
            sum(if(species_id = predicted_species_id, 1, 0)) number_of_correct_predictions,
            sum(if(species_id = predicted_species_id, 0, 1)) number_of_incorrect_predictions
        """)
    )
    return


if __name__ == "__main__":
    app.run()
