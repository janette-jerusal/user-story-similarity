# app.py — minimal, loud startup + your similarity tool
import io
import sys
import traceback
from datetime import datetime, timezone

print("=== APP IMPORT START ===", flush=True)

try:
    import numpy as np
    import pandas as pd
    import streamlit as st
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    print("=== IMPORTS OK ===", flush=True)

    st.set_page_config(page_title="User Story Similarity", layout="wide")

    BUILD_TS = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    st.title("USCAP")
    st.subheader("User Story Comparison Analysis Program")
    st.caption(f"Build: {BUILD_TS} UTC • One-to-one pairs only")

    @st.cache_data(show_spinner=False)
    def read_table_from_bytes(name: str, uploaded_bytes: bytes) -> pd.DataFrame:
        bio = io.BytesIO(uploaded_bytes)
        lower = name.lower()

        # Try Excel first if extension looks like Excel
        if lower.endswith((".xlsx", ".xls")):
            try:
                return pd.read_excel(bio)
            except Exception:
                bio.seek(0)

        # Try CSV
        try:
            bio.seek(0)
            return pd.read_csv(bio)
        except Exception:
            # Last resort: try Excel anyway
            bio.seek(0)
            return pd.read_excel(bio)

    def guess_columns(df: pd.DataFrame):
        cols = [str(c).strip() for c in df.columns]
        norm = {c: c.lower().strip() for c in cols}

        id_aliases = {"id", "story id", "user story id", "issue id", "key", "ticket", "task id"}
        desc_aliases = {"description", "user story", "story", "summary", "title", "details", "acceptance criteria"}

        id_candidates = [c for c in cols if norm[c] in id_aliases]
        desc_candidates = [c for c in cols if norm[c] in desc_aliases]

        id_col = id_candidates[0] if id_candidates else cols[0]
        desc_col = desc_candidates[0] if desc_candidates else (cols[1] if len(cols) > 1 else cols[0])
        return id_col, desc_col

    def clean_df(df: pd.DataFrame, id_col: str, desc_col: str) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        df[id_col] = df[id_col].astype(str).str.strip()
        df[desc_col] = (
            df[desc_col]
            .astype(str)
            .fillna("")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        df = df[(df[id_col] != "") & (df[desc_col] != "")]
        df = df.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)
        return df

    def build_vectorizer(ngram_min=1, ngram_max=2, min_df=1, max_df=1.0) -> TfidfVectorizer:
        if ngram_max < ngram_min:
            ngram_max = ngram_min
        return TfidfVectorizer(
            stop_words="english",
            ngram_range=(int(ngram_min), int(ngram_max)),
            min_df=int(min_df),
            max_df=float(max_df),
        )

    def upper_triangle_long(sim_matrix: np.ndarray, ids) -> pd.DataFrame:
        sim_df = pd.DataFrame(sim_matrix, index=ids, columns=ids)
        upper_only = sim_df.where(np.triu(np.ones(sim_df.shape, dtype=bool), k=1))
        return (
            upper_only.stack()
            .reset_index()
            .rename(columns={"level_0": "ID_A", "level_1": "ID_B", 0: "similarity"})
        )

    def cross_long(sim_matrix: np.ndarray, ids_a, ids_b) -> pd.DataFrame:
        sim_df = pd.DataFrame(sim_matrix, index=ids_a, columns=ids_b)
        return (
            sim_df.stack()
            .reset_index()
            .rename(columns={"level_0": "ID_A", "level_1": "ID_B", 0: "similarity"})
        )

    # ---- UI ----
    mode = st.radio(
        "Comparison mode",
        ["One file (all vs all, once each)", "Two files (A vs B)"],
        horizontal=True,
    )

    with st.sidebar:
        st.header("Filters")
        threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.30, 0.01)
        topk_per_A = st.number_input("Top-K per ID_A (0 = no limit)", min_value=0, value=0, step=1)
        sort_desc = st.checkbox("Sort by similarity (desc)", value=True)
        show_preview_rows = st.number_input("Preview rows", min_value=10, max_value=200, value=50, step=10)

    with st.expander("Advanced settings", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            ngram_min = st.number_input("Min n-gram", 1, 5, 1, 1)
            ngram_max = st.number_input("Max n-gram", 1, 5, 2, 1)
        with c2:
            min_df = st.number_input("min_df", 1, value=1, step=1)
        with c3:
            max_df = st.slider("max_df", 0.1, 1.0, 1.0, 0.05)

    if mode.startswith("One file"):
        file1 = st.file_uploader("Upload file (Excel/CSV)", type=["xlsx", "xls", "csv"])
        if file1 is not None:
            df_raw = read_table_from_bytes(file1.name, file1.getvalue())
            st.dataframe(df_raw.head(10), use_container_width=True)

            id_guess, desc_guess = guess_columns(df_raw)
            c1, c2 = st.columns(2)
            with c1:
                id_col = st.selectbox("ID column", df_raw.columns, index=list(df_raw.columns).index(id_guess))
            with c2:
                desc_col = st.selectbox("Description column", df_raw.columns, index=list(df_raw.columns).index(desc_guess))

            df = clean_df(df_raw, id_col, desc_col)
            if st.button("Compute similarities"):
                vec = build_vectorizer(ngram_min, ngram_max, min_df, max_df)
                X = vec.fit_transform(df[desc_col].tolist())
                if X.shape[0] < 2:
                    st.warning("Need at least 2 rows.")
                else:
                    sim = cosine_similarity(X, X)
                    pairs = upper_triangle_long(sim, df[id_col].tolist())
                    pairs = pairs[pairs["similarity"] >= threshold]

                    if sort_desc:
                        pairs = pairs.sort_values("similarity", ascending=False)

                    if topk_per_A > 0:
                        pairs = pairs.groupby("ID_A", as_index=False, sort=False).head(int(topk_per_A))

                    st.dataframe(pairs.head(int(show_preview_rows)), use_container_width=True)

    else:
        fileA = st.file_uploader("Upload File A", type=["xlsx", "xls", "csv"])
        fileB = st.file_uploader("Upload File B", type=["xlsx", "xls", "csv"])

        if fileA is not None and fileB is not None:
            dfA_raw = read_table_from_bytes(fileA.name, fileA.getvalue())
            dfB_raw = read_table_from_bytes(fileB.name, fileB.getvalue())

            st.write("File A")
            st.dataframe(dfA_raw.head(8), use_container_width=True)
            st.write("File B")
            st.dataframe(dfB_raw.head(8), use_container_width=True)

            idA_guess, descA_guess = guess_columns(dfA_raw)
            idB_guess, descB_guess = guess_columns(dfB_raw)

            c1, c2 = st.columns(2)
            with c1:
                idA = st.selectbox("A: ID col", dfA_raw.columns, index=list(dfA_raw.columns).index(idA_guess))
                descA = st.selectbox("A: Desc col", dfA_raw.columns, index=list(dfA_raw.columns).index(descA_guess))
            with c2:
                idB = st.selectbox("B: ID col", dfB_raw.columns, index=list(dfB_raw.columns).index(idB_guess))
                descB = st.selectbox("B: Desc col", dfB_raw.columns, index=list(dfB_raw.columns).index(descB_guess))

            dfA = clean_df(dfA_raw, idA, descA)
            dfB = clean_df(dfB_raw, idB, descB)

            if st.button("Compute A vs B"):
                vec = build_vectorizer(ngram_min, ngram_max, min_df, max_df)
                vec.fit(pd.concat([dfA[descA], dfB[descB]], ignore_index=True).tolist())
                XA = vec.transform(dfA[descA].tolist())
                XB = vec.transform(dfB[descB].tolist())

                sim = cosine_similarity(XA, XB)
                pairs = cross_long(sim, dfA[idA].tolist(), dfB[idB].tolist())
                pairs = pairs[pairs["similarity"] >= threshold]

                if sort_desc:
                    pairs = pairs.sort_values("similarity", ascending=False)

                if topk_per_A > 0:
                    pairs = pairs.groupby("ID_A", as_index=False, sort=False).head(int(topk_per_A))

                st.dataframe(pairs.head(int(show_preview_rows)), use_container_width=True)

    print("=== APP RUN END (should not normally hit) ===", flush=True)

except Exception as e:
    print("=== APP CRASHED DURING IMPORT/RUN ===", flush=True)
    traceback.print_exc()
    # If Streamlit imported, show the error on-page too
    try:
        import streamlit as st
        st.error("App crashed on startup. See logs for traceback.")
        st.code("".join(traceback.format_exception(*sys.exc_info())))
    except Exception:
        pass
    raise

