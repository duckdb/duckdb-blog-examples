import marimo

__generated_with = "0.13.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import plotly.graph_objects as go

    return go, mo


@app.cell
def _(mo):
    import os

    process_data = not os.path.exists("./data/text_analytics.duckdb")

    mo.plain_text(f"Process data: {process_data}")
    return (process_data,)


@app.cell
def _():
    import duckdb

    conn = duckdb.connect("./data/text_analytics.duckdb")
    return (conn,)


@app.cell
def _(conn, process_data):
    def load_data_from_hf(duckdb_conn):
        # save data from hf
        duckdb_conn.sql("drop table if exists text_emotions")

        from_hf_rel = duckdb_conn.read_parquet(
            "hf://datasets/dair-ai/emotion/unsplit/train-00000-of-00001.parquet",
            file_row_number=True,
        )
        from_hf_rel = from_hf_rel.select("""
            text,
            label as emotion_id,
            file_row_number as text_id
        """)
        from_hf_rel.to_table("text_emotions")

    if process_data:
        load_data_from_hf(conn)
    return


@app.cell
def _(conn):
    emotion_colors = {
        "sadness": "#a6cee3",  # soft blue
        "joy": "#fdbf6f",  # light orange
        "love": "#fbb4ae",  # pastel pink
        "anger": "#fb8072",  # muted red
        "fear": "#cab2d6",  # soft lavender
        "surprise": "#b3de69",  # gentle green
    }

    def get_emotion_color(emotion: str) -> str:
        return emotion_colors.get(emotion)

    conn.create_function("get_emotion_color", get_emotion_color)
    return (emotion_colors,)


@app.cell
def _(conn):
    def load_emotion_ref(duckdb_conn):
        # save emotions to reference table
        # sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5)

        duckdb_conn.sql("drop table if exists emotion_ref")

        emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

        from_labels_rel = duckdb_conn.values([emotion_labels])
        from_labels_rel = from_labels_rel.select(f"""
            unnest(col0) as emotion,
            generate_subscripts(col0, 1) - 1 as emotion_id,
            get_emotion_color(emotion) as emotion_color
        """)
        from_labels_rel.to_table("emotion_ref")

    load_emotion_ref(conn)
    return


@app.cell
def _(conn):
    def create_view(duckdb_conn):
        text_emotions_rel = duckdb_conn.table("text_emotions")
        emotion_ref_rel = duckdb_conn.table("emotion_ref")
        text_emotions_rel.join(emotion_ref_rel, condition="emotion_id").to_view(
            "text_emotions_v", replace=True
        )

    create_view(conn)
    return


@app.cell
def _(conn, emotion_colors):
    import plotly.express as px

    px.bar(
        (
            conn.view("text_emotions_v")
            .count(
                column="text",
                groups="emotion",
                projected_columns="count(text) as number_of_records, emotion",
            )
            .order("emotion")
        ),
        x="emotion",
        y="number_of_records",
        title="Emotions in text",
        labels={"number_of_records": "Number of records", "emotion": "Emotion"},
        color="emotion",
        color_discrete_map=emotion_colors,
    ).update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, showticklabels=True),
        yaxis=dict(showgrid=False, showticklabels=True),
    )
    return (px,)


@app.cell
def _(mo):
    mo.md(r"""# Keyword""")
    return


@app.cell
def _(conn):
    view_rel = conn.view("text_emotions_v")
    view_rel = view_rel.filter("text ilike '%excited to learn%'")
    view_rel.select("""
        emotion,
        substring(
            text,
            position('excited to learn' in text),
            position('excited to learn' in text) + len('excited to learn') - 2
        ) as substring_text 
    """)
    return


@app.cell
def _(conn):
    def create_text_tokens(duckdb_conn):
        conn.sql("drop table if exists text_emotion_tokens")

        text_emotions_tokenized_rel = duckdb_conn.view("text_emotions_v").select("""
            text_id,
            emotion,
            regexp_split_to_table(text, '\\W+') as token
        """)

        english_stopwords_rel = duckdb_conn.read_csv(
            "https://raw.githubusercontent.com/stopwords-iso/stopwords-en/refs/heads/master/stopwords-en.txt",
            header=False,
        ).select("column0 as token")

        conn.sql("drop table if exists english_stopwords")
        english_stopwords_rel.to_table("english_stopwords")

        text_emotions_tokenized_rel.join(
            conn.table("english_stopwords"),
            condition="token",
            how="anti",
        ).to_table("text_emotion_tokens")

    create_text_tokens(conn)
    return


@app.cell
def _(conn):
    # jaccard

    text_token_rel = conn.table("text_emotion_tokens").select(
        "token, emotion, jaccard(token, 'learn') as jaccard_score"
    )

    text_token_rel = text_token_rel.max(
        "jaccard_score", groups="emotion, token", projected_columns="emotion, token"
    )

    text_token_rel.order("3 desc").limit(10)

    return


@app.cell
def _(conn, px):
    count_rel = (
        conn.table("text_emotion_tokens")
        .count(
            "*",
            groups="emotion, token",
            projected_columns="count(*) as number_of_occurrences, emotion, token",
        )
        .select("*, log(number_of_occurrences) as log_no")
    )

    px.scatter(
        conn.sql(f"""
            {count_rel.sql_query()}
            qualify row_number() over (partition by emotion order by number_of_occurrences desc) <= 7 
        """).order("token, emotion"),
        x="number_of_occurrences",
        y="token",
        color="token",
        color_discrete_sequence=px.colors.sequential.Plasma,
        size="log_no",
        facet_col="emotion",
        labels={"number_of_occurrences": "Occurrences", "token": ""},
        title="Most frequently used words, per emotion",
    ).update_layout(
        showlegend=False,
        yaxis=dict(tickfont=dict(size=14), dtick="1"),
        xaxis1=dict(showgrid=False, zeroline=False),
        xaxis2=dict(showgrid=False, zeroline=False),
        xaxis3=dict(showgrid=False, zeroline=False),
        xaxis4=dict(showgrid=False, zeroline=False),
        xaxis5=dict(showgrid=False, zeroline=False),
        xaxis6=dict(showgrid=False, zeroline=False),
    )
    return


@app.cell
def _(mo):
    mo.md(r"""# FTS""")
    return


@app.cell
def _(conn):
    conn.sql("install fts")
    conn.sql("load fts")
    return


@app.cell
def _(conn, px):
    px.scatter(
        conn.sql("""
            select *, log(number_of_occurrences) as log_no
                from (
                select 
                    emotion,
                    stem(token, 'english') as token,
                    count(*) as number_of_occurrences
                from text_emotion_tokens
                group by all
            )
            qualify row_number() over (partition by emotion order by number_of_occurrences desc) <= 7 
        """).order("token, emotion"),
        x="number_of_occurrences",
        y="token",
        color="token",
        color_discrete_sequence=px.colors.sequential.Plasma,
        size="log_no",
        facet_col="emotion",
        labels={"number_of_occurrences": "Occurrences"},
        title="Most frequently used word stem, per emotion",
    ).update_layout(
        showlegend=False,
        yaxis=dict(tickfont=dict(size=14), dtick="1"),
        xaxis1=dict(showgrid=False, zeroline=False),
        xaxis2=dict(showgrid=False, zeroline=False),
        xaxis3=dict(showgrid=False, zeroline=False),
        xaxis4=dict(showgrid=False, zeroline=False),
        xaxis5=dict(showgrid=False, zeroline=False),
        xaxis6=dict(showgrid=False, zeroline=False),
    )
    return


@app.cell
def _(conn):
    def create_fts_index(duckdb_conn):
        duckdb_conn.sql("""
            PRAGMA create_fts_index(
                "text_emotions", 
                text_id, 
                "text",
                stemmer='english',
                stopwords='english_stopwords', 
                ignore='(\\.|[^a-z])+',
                strip_accents=1, 
                lower=1, 
                overwrite=1
            )
        """)

    create_fts_index(conn)
    return


@app.cell
def _(conn, go):
    df_fts = (
        conn.view("text_emotions_v")
        .select("""
            text_id,
            emotion,
            text,
            emotion_color,
            fts_main_text_emotions.match_bm25(
                text_id,
                'excited to learn',
                fields := 'text'
            )::decimal(3, 2) as bm25_score
        """)
        .order("bm25_score desc, text_id")
        .limit(10)
    ).df()

    fig_fts = go.Figure(
        data=[
            go.Table(
                columnwidth=[20, 100, 20],
                header=dict(
                    values=["<b>Emotion</b>", "<b>Text</b>", "<b>BM25 Score</b>"],
                    line_color="white",
                    fill_color="white",
                    align="center",
                    font=dict(color="black", size=12),
                ),
                cells=dict(
                    values=[df_fts.emotion, df_fts.text, df_fts.bm25_score],
                    fill_color=[df_fts.emotion_color],
                    align=["left", "center"],
                    font_size=14,
                    height=30,
                ),
            )
        ]
    )
    fig_fts.update_layout(margin=dict(l=20, r=20, t=20, b=20))

    fig_fts.write_html("./__marimo__/fts.html")

    fig_fts

    return


@app.cell
def _(mo):
    mo.md(r"""# Embeddings""")
    return


@app.cell
def _(conn):
    conn.sql("install vss")
    conn.sql("load vss")
    return


@app.cell
def _():
    from sentence_transformers import SentenceTransformer

    def download_model():
        initial_model = SentenceTransformer("all-MiniLM-L6-v2")
        initial_model.save("./sentence_transformer")

    # if getting 429, wait for 1-2 minutes and run again
    download_model()
    return (SentenceTransformer,)


@app.cell
def _(SentenceTransformer, conn):
    model = SentenceTransformer("./sentence_transformer")

    def get_text_embedding_list(list_text: list[str]):
        """
        Return the list of normalized vector embeddings for the list of text provided as input.
        """
        return model.encode(list_text, normalize_embeddings=True)

    # conn.remove_function("get_text_embedding_list")
    conn.create_function(
        "get_text_embedding_list", get_text_embedding_list, return_type="FLOAT[384][]"
    )
    return


@app.cell
def _(conn, mo, process_data):
    from datetime import datetime

    def create_text_embeddings(duckdb_conn):
        conn.sql("drop table if exists text_emotion_embeddings")
        duckdb_conn.sql(
            "create table text_emotion_embeddings (text_id integer, text_embedding FLOAT[384])"
        )

        batch_size = 15000

        num_batches = int(
            duckdb_conn.table("text_emotions")
            .aggregate(f"ceil(count(*)/{batch_size})")
            .fetchone()[0]
        )

        for i in range(num_batches):
            st = datetime.now()
            selection_query = (
                duckdb_conn.table("text_emotions")
                .order("text_id")
                .limit(batch_size, offset=batch_size * i)
                .select("*")
            )
            (
                selection_query.aggregate("""
                    array_agg(text) as text_list,
                    array_agg(text_id) as id_list,
                    get_text_embedding_list(text_list) as text_emb_list
                """).select("""
                    unnest(id_list) as text_id,
                    unnest(text_emb_list) as text_embedding
                """)
            ).insert_into("text_emotion_embeddings")

            with mo.redirect_stdout():
                print(
                    f"done {i} from {num_batches} in {(datetime.now() - st).total_seconds()} seconds"
                )

    if process_data:
        create_text_embeddings(conn)
    return


@app.cell
def _(conn, go):
    df_vss = (
        conn.view("text_emotions_v")
        .join(conn.table("text_emotion_embeddings"), condition="text_id")
        .select("""
                text, 
                emotion,
                emotion_color,
                array_cosine_distance(
                    text_embedding,
                    get_text_embedding_list(['excited to learn'])[1]
                )::decimal(3,2) as cosine_distance_score
            """)
        .order("cosine_distance_score asc")
        .limit(10)
    ).df()

    fig_vss = go.Figure(
        data=[
            go.Table(
                columnwidth=[20, 100, 20],
                header=dict(
                    values=[
                        "<b>Emotion</b>",
                        "<b>Text</b>",
                        "<b>Cosine Distance Score</b>",
                    ],
                    line_color="white",
                    fill_color="white",
                    align="center",
                    font=dict(color="black", size=12),
                ),
                cells=dict(
                    values=[df_vss.emotion, df_vss.text, df_vss.cosine_distance_score],
                    fill_color=[df_vss.emotion_color],
                    align=["left", "center"],
                    font_size=14,
                    height=30,
                ),
            )
        ]
    )

    fig_vss.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    fig_vss.write_html("./__marimo__/vss.html")

    fig_vss
    return


@app.cell
def _(conn, go):
    # hybrid search

    df_hybrid = (
        conn.view("text_emotions_v")
        .join(conn.table("text_emotion_embeddings"), condition="text_id")
        .select("""
            text,
            emotion,
            emotion_color,
            if(emotion = 'joy' and contains(text, 'excited to learn'), 1, 0) exact_match_score,
            1 - array_cosine_distance(
                text_embedding,
                get_text_embedding_list(['excited to learn'])[1]
            )::decimal(3,2) as cosine_distance_score,
            fts_main_text_emotions.match_bm25(
                    text_id,
                    'excited to learn'
            )::decimal(3,2) AS bm25_score
        """)
        .select("""
            *,
            (cosine_distance_score - min(cosine_distance_score) over ()) / NULLIF((max(cosine_distance_score) over () - min(cosine_distance_score) over ()), 0) AS norm_cosine_distance_score,
            (bm25_score - min(bm25_score) over ()) / NULLIF((max(bm25_score) over () - min(bm25_score) over ()), 0) AS norm_bm25_score,
            if(exact_match_score = 1, exact_match_score,
            cast(
                0.3 * coalesce(norm_bm25_score, 0) + 0.7 * coalesce(norm_cosine_distance_score, 0)
                as
                decimal(3, 2)
            )) as hybrid_score
        """)
        .select("""
            *,    

        """)
        .select("emotion, text, emotion_color, hybrid_score")
        .order("hybrid_score desc")
        .limit(10)
    ).df()

    fig_hybrid = go.Figure(
        data=[
            go.Table(
                columnwidth=[20, 100, 20],
                header=dict(
                    values=["<b>Emotion</b>", "<b>Text</b>", "<b>Hybrid Score</b>"],
                    line_color="white",
                    fill_color="white",
                    align="center",
                    font=dict(color="black", size=12),
                ),
                cells=dict(
                    values=[df_hybrid.emotion, df_hybrid.text, df_hybrid.hybrid_score],
                    fill_color=[df_hybrid.emotion_color],
                    align=["left", "center"],
                    font_size=14,
                    height=30,
                ),
            )
        ]
    )

    fig_hybrid.update_layout(margin=dict(l=20, r=20, t=20, b=20))

    fig_hybrid.write_html("./__marimo__/hybrid.html")

    fig_hybrid

    return


if __name__ == "__main__":
    app.run()
