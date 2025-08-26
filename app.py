# app.py
# Streamlit app for User Story Similarity
# - Ensures each pair is compared ONCE (A→B without B→A) for all-vs-all mode
# - For cross-file (A vs B), pairs are naturally unique
#
# Run: streamlit run app.py

import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------
# Helpers
# -------------------------
def read_table(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    elif name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        # Try excel first then csv as fallback
        try:
            return pd.read_excel(file)
        except Exception:
            file.seek(0)
            return pd.read_csv(file)


def guess_columns(df: pd.DataFrame):
    cols = [c for c in df.columns]
    # Heuristics for ID and Description columns
    id_candidates = [c for c in cols if c.strip().lower() in ["id", "story id", "user story id", "issue id", "key"]]
    desc_candidates = [c for c in cols if c.strip().lower() in ["description", "user story", "story", "summary", "title"]]

    id_col = id_candidates[0] if id_candidates else cols[0]
    desc_col = desc_candidates[0] if desc_candidates else cols[min(1, len(cols) - 1)]
    return id_col, desc_col


def build_vectorizer(ngram_min=1, ngram_max=2, min_df=1, max_df=1.0):
    return TfidfVectorizer(
        stop_words="english",
        ngram_range=(ngram_min, ngram_max),
        min_df=min_df,
        max_df=max_df
    )


def upper_triangle_long(sim_matrix: np.ndarray, ids: pd.Series | list) -> pd.DataFrame:
    """Return long-form pairs from ONLY the upper triangle (no diagonal)."""
    sim_df = pd.DataFrame(sim_matrix, index=ids, columns=ids)
    upper_only = sim_df.where(np.triu(np.ones(sim_df.shape, dtype=bool), k=1))
    out = (
        upper_only
        .stack()
        .reset_index()
        .rename(columns={"level_0": "ID_A", "level_1": "ID_B", 0: "similarity"})
    )
    return out


def cross_long(sim_matrix: np.ndarray, ids_a, ids_b) -> pd.DataFrame:
    """Return long-form pairs for A×B matrix."""
    sim_df = pd.DataFrame(sim_matrix, index=ids_a, columns=ids_b)
    out = (
        sim_df
        .stack()
        .reset_index()
        .rename(columns={"level_0": "ID_A", "level_1": "ID_B", 0: "similarity"})
    )
    return out


def downloadable_excel(df_pairs: pd.DataFrame, sheet_name="pairs_once") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df_pairs.to_excel(writer, index=False, sheet_name=sheet_name)
    buf.seek(0)
    return buf.read()


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="User Story Similarity", layout="wide")
st.title("🔎 User Story Similarity (TF-IDF • Cosine)")

mode = st.radio("Comparison Mode", ["One file (all-vs-all)", "Two files (A vs B)"], horizontal=True)

with st.expander("Vectorizer Settings (optional)"):
    col_v1, col_v2, col_v3 = st.columns(3)
    with col_v1:
        ngram_min = st.number_input("Min n-gram", min_value=1, max_value=3, value=1, step=1)
        ngram_max = st.number_input("Max n-gram", min_value=1, max_value=3, value=2, step=1)
    with col_v2:
        min_df = st.number_input("min_df", min_value=1, value=1, step=1)
    with col_v3:
        max_df = st.slider("max_df", min_value=0.1, max_value=1.0, value=1.0, step=0.05)

with st.sidebar:
    st.header("Filters")
    threshold = st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    topk_per_A = st.number_input("Top-K per ID_A (0 = no cap)", min_value=0, value=0, step=1)
    sort_desc = st.checkbox("Sort by similarity (desc)", value=True)
    show_preview_rows = st.number_input("Preview rows", min_value=10, max_value=200, value=50, step=10)

if mode == "One file (all-vs-all)":
    file1 = st.file_uploader("Upload a file (Excel or CSV) with ID and Description columns", type=["xlsx", "xls", "csv"])
    if file1 is not None:
        df = read_table(file1)
        st.write("**Data preview:**")
        st.dataframe(df.head(10), use_container_width=True)

        id_guess, desc_guess = guess_columns(df)
        c1, c2 = st.columns(2)
        with c1:
            id_col = st.selectbox("ID column", df.columns, index=list(df.columns).index(id_guess))
        with c2:
            desc_col = st.selectbox("Description column", df.columns, index=list(df.columns).index(desc_guess))

        # Clean
        df = df.copy()
        df[id_col] = df[id_col].astype(str)
        df[desc_col] = df[desc_col].fillna("").astype(str)

        if st.button("Compute Similarities (All-vs-All)"):
            with st.spinner("Vectorizing and computing similarities..."):
                vec = build_vectorizer(ngram_min, ngram_max, min_df, max_df)
                X = vec.fit_transform(df[desc_col].tolist())
                sim = cosine_similarity(X, X)

                pairs = upper_triangle_long(sim, ids=df[id_col].tolist())

                # Apply threshold
                if threshold > 0:
                    pairs = pairs[pairs["similarity"] >= threshold]

                # Top-K per ID_A
                if topk_per_A > 0:
                    pairs = (
                        pairs.sort_values("similarity", ascending=not sort_desc)
                             .groupby("ID_A", as_index=False)
                             .head(topk_per_A)
                    )
                else:
                    pairs = pairs.sort_values("similarity", ascending=not sort_desc) if sort_desc else pairs

            st.success(f"Done. {len(pairs):,} unique pairs (A→B only).")
            st.dataframe(pairs.head(show_preview_rows), use_container_width=True)

            cdl1, cdl2 = st.columns(2)
            with cdl1:
                csv_bytes = pairs.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="similarity_pairs_once.csv", mime="text/csv")
            with cdl2:
                xlsx_bytes = downloadable_excel(pairs)
                st.download_button("⬇️ Download Excel", data=xlsx_bytes, file_name="similarity_pairs_once.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    fileA = st.file_uploader("Upload File A (Excel or CSV)", type=["xlsx", "xls", "csv"], key="A")
    fileB = st.file_uploader("Upload File B (Excel or CSV)", type=["xlsx", "xls", "csv"], key="B")

    if fileA is not None and fileB is not None:
        dfA = read_table(fileA)
        dfB = read_table(fileB)

        st.write("**Preview A:**")
        st.dataframe(dfA.head(8), use_container_width=True)
        st.write("**Preview B:**")
        st.dataframe(dfB.head(8), use_container_width=True)

        idA_guess, descA_guess = guess_columns(dfA)
        idB_guess, descB_guess = guess_columns(dfB)

        cc1, cc2 = st.columns(2)
        with cc1:
            idA = st.selectbox("File A — ID column", dfA.columns, index=list(dfA.columns).index(idA_guess))
            descA = st.selectbox("File A — Description column", dfA.columns, index=list(dfA.columns).index(descA_guess))
        with cc2:
            idB = st.selectbox("File B — ID column", dfB.columns, index=list(dfB.columns).index(idB_guess))
            descB = st.selectbox("File B — Description column", dfB.columns, index=list(dfB.columns).index(descB_guess))

        # Clean
        dfA = dfA.copy()
        dfB = dfB.copy()
        dfA[idA] = dfA[idA].astype(str)
        dfB[idB] = dfB[idB].astype(str)
        dfA[descA] = dfA[descA].fillna("").astype(str)
        dfB[descB] = dfB[descB].fillna("").astype(str)

        if st.button("Compute Similarities (A vs B)"):
            with st.spinner("Vectorizing and computing cross similarities..."):
                vec = build_vectorizer(ngram_min, ngram_max, min_df, max_df)
                # Fit on combined corpus to align feature space across A and B
                combined = pd.concat([dfA[descA], dfB[descB]], ignore_index=True)
                vec.fit(combined.tolist())
                XA = vec.transform(dfA[descA].tolist())
                XB = vec.transform(dfB[descB].tolist())

                sim = cosine_similarity(XA, XB)
                pairs = cross_long(sim, ids_a=dfA[idA].tolist(), ids_b=dfB[idB].tolist())

                # Threshold
                if threshold > 0:
                    pairs = pairs[pairs["similarity"] >= threshold]

                # Top-K per A
                if topk_per_A > 0:
                    pairs = (
                        pairs.sort_values("similarity", ascending=not sort_desc)
                             .groupby("ID_A", as_index=False)
                             .head(topk_per_A)
                    )
                else:
                    pairs = pairs.sort_values("similarity", ascending=not sort_desc) if sort_desc else pairs

            st.success(f"Done. {len(pairs):,} A×B pairs.")
            st.dataframe(pairs.head(show_preview_rows), use_container_width=True)

            cdl1, cdl2 = st.columns(2)
            with cdl1:
                csv_bytes = pairs.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="similarity_pairs_A_vs_B.csv", mime="text/csv")
            with cdl2:
                xlsx_bytes = downloadable_excel(pairs, sheet_name="pairs_A_vs_B")
                st.download_button("⬇️ Download Excel", data=xlsx_bytes, file_name="similarity_pairs_A_vs_B.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


