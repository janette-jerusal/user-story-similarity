# app.py
# User Story Similarity — simple, friendly UI
# - One-file mode: compares each pair ONCE (upper triangle, no self-pairs)
# - Two-file mode: A vs B once (rectangular)
# - Outputs: ID_A, Desc_A, ID_B, Desc_B, similarity

import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Small helpers
# -------------------------
def read_table(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    elif name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        # Try Excel first, then CSV
        try:
            return pd.read_excel(file)
        except Exception:
            file.seek(0)
            return pd.read_csv(file)

def guess_columns(df: pd.DataFrame):
    cols = list(df.columns)
    id_candidates = [c for c in cols if c.strip().lower() in ["id","story id","user story id","issue id","key"]]
    desc_candidates = [c for c in cols if c.strip().lower() in ["description","user story","story","summary","title"]]
    id_col = id_candidates[0] if id_candidates else cols[0]
    desc_col = desc_candidates[0] if desc_candidates else cols[min(1, len(cols)-1)]
    return id_col, desc_col

def build_vectorizer(ngram_min=1, ngram_max=2, min_df=1, max_df=1.0):
    return TfidfVectorizer(
        stop_words="english",
        ngram_range=(ngram_min, ngram_max),
        min_df=min_df,
        max_df=max_df
    )

def upper_triangle_long(sim_matrix: np.ndarray, ids) -> pd.DataFrame:
    sim_df = pd.DataFrame(sim_matrix, index=ids, columns=ids)
    upper_only = sim_df.where(np.triu(np.ones(sim_df.shape, dtype=bool), k=1))
    out = (upper_only.stack().reset_index()
           .rename(columns={"level_0":"ID_A","level_1":"ID_B",0:"similarity"}))
    return out

def cross_long(sim_matrix: np.ndarray, ids_a, ids_b) -> pd.DataFrame:
    sim_df = pd.DataFrame(sim_matrix, index=ids_a, columns=ids_b)
    out = (sim_df.stack().reset_index()
           .rename(columns={"level_0":"ID_A","level_1":"ID_B",0:"similarity"}))
    return out

def downloadable_excel(df_pairs: pd.DataFrame, sheet_name="pairs") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        df_pairs.to_excel(w, index=False, sheet_name=sheet_name)
    buf.seek(0)
    return buf.read()

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="User Story Similarity", layout="wide")
st.title("🔎 User Story Similarity (TF-IDF · Cosine)")
st.caption(f"Build: {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC • One-to-one pairs only")

mode = st.radio(
    "Comparison mode",
    ["One file (all vs all, once each)", "Two files (A vs B)"],
    horizontal=True,
    help="Choose whether to compare stories within a single file (every pair once) or compare stories from File A against File B."
)

st.info(
    "Each pair is compared **once** only.\n"
    "• **One file**: uses the upper triangle → no A↔B duplicates, no self-pairs.\n"
    "• **Two files**: A vs B pairs only (no B vs A)."
)

with st.sidebar:
    st.header("Filters")
    threshold = st.slider(
        "Similarity threshold",
        0.0, 1.0, 0.30, 0.01,
        help="Hide pairs below this cosine similarity. Start around 0.30–0.50; increase to see only closer matches."
    )
    topk_per_A = st.number_input(
        "Top-K per ID_A (0 = no limit)",
        min_value=0, value=0, step=1,
        help="Limit how many best matches you keep for each source story (ID_A). Set 0 to keep all that pass the threshold."
    )
    sort_desc = st.checkbox(
        "Sort by similarity (desc)",
        value=True,
        help="When enabled, results are sorted from most similar to least."
    )
    show_preview_rows = st.number_input(
        "Preview rows",
        min_value=10, max_value=200, value=50, step=10,
        help="How many rows to preview in the on-screen table (download contains all)."
    )

with st.expander("Advanced settings (optional)", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        ngram_min = st.number_input("Min n-gram", 1, 3, 1, 1, help="Smallest token span for TF-IDF (1 = unigrams).")
        ngram_max = st.number_input("Max n-gram", 1, 3, 2, 1, help="Largest token span for TF-IDF (2 = capture short phrases).")
    with c2:
        min_df = st.number_input("min_df", 1, value=1, step=1, help="Ignore terms that appear in fewer than this many documents.")
    with c3:
        max_df = st.slider("max_df", 0.1, 1.0, 1.0, 0.05, help="Ignore terms that appear in more than this fraction of documents.")

# -------------------------
# ONE FILE
# -------------------------
if mode.startswith("One file"):
    file1 = st.file_uploader(
        "Upload a file (Excel/CSV) with **ID** and **Description** columns",
        type=["xlsx","xls","csv"],
        help="Typical columns: ID, Description (or Story, Summary, Title)."
    )

    if file1 is not None:
        df = read_table(file1)
        st.write("**Data preview**")
        st.dataframe(df.head(10), use_container_width=True)

        # Column selection with guesses
        id_guess, desc_guess = guess_columns(df)
        c1, c2 = st.columns(2)
        with c1:
            id_col = st.selectbox("ID column", df.columns, index=list(df.columns).index(id_guess),
                                  help="Unique identifier for each story (e.g., ID, Key).")
        with c2:
            desc_col = st.selectbox("Description column", df.columns, index=list(df.columns).index(desc_guess),
                                    help="Text describing the story; this is what gets vectorized.")

        # Clean + validate
        df = df.copy()
        df[id_col] = df[id_col].astype(str)
        df[desc_col] = df[desc_col].fillna("").astype(str)

        if df[desc_col].str.strip().eq("").all():
            st.warning("All descriptions are empty after cleaning. Please select the correct description column.")
        else:
            if st.button("Compute similarities (One file)"):
                with st.spinner("Vectorizing and computing…"):
                    vec = build_vectorizer(ngram_min, ngram_max, min_df, max_df)
                    X = vec.fit_transform(df[desc_col].tolist())
                    sim = cosine_similarity(X, X)

                    pairs = upper_triangle_long(sim, ids=df[id_col].tolist())

                    # Map both descriptions
                    id_to_desc = dict(zip(df[id_col], df[desc_col]))
                    pairs["Desc_A"] = pairs["ID_A"].map(id_to_desc)
                    pairs["Desc_B"] = pairs["ID_B"].map(id_to_desc)

                    # Reorder + filter
                    pairs = pairs[["ID_A","Desc_A","ID_B","Desc_B","similarity"]]
                    if threshold > 0:
                        pairs = pairs[pairs["similarity"] >= threshold]

                    if topk_per_A > 0:
                        pairs = (pairs.sort_values("similarity", ascending=not sort_desc)
                                       .groupby("ID_A", as_index=False).head(topk_per_A))
                    elif sort_desc:
                        pairs = pairs.sort_values("similarity", ascending=False)

                st.success(f"Done. {len(pairs):,} unique pairs (A→B only).")
                st.dataframe(pairs.head(show_preview_rows), use_container_width=True)

                c1, c2 = st.columns(2)
                with c1:
                    st.download_button(
                        "⬇️ Download CSV",
                        data=pairs.to_csv(index=False).encode("utf-8"),
                        file_name="similarity_pairs_once.csv",
                        mime="text/csv"
                    )
                with c2:
                    st.download_button(
                        "⬇️ Download Excel",
                        data=downloadable_excel(pairs, sheet_name="pairs_once"),
                        file_name="similarity_pairs_once.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

# -------------------------
# TWO FILES
# -------------------------
else:
    fileA = st.file_uploader("Upload File A (Excel/CSV)", type=["xlsx","xls","csv"],
                             help="Source set A — each A story will be compared to every B story.", key="A")
    fileB = st.file_uploader("Upload File B (Excel/CSV)", type=["xlsx","xls","csv"],
                             help="Target set B — results list A vs B pairs once.", key="B")

    if fileA is not None and fileB is not None:
        dfA = read_table(fileA)
        dfB = read_table(fileB)

        st.write("**Preview A**"); st.dataframe(dfA.head(8), use_container_width=True)
        st.write("**Preview B**"); st.dataframe(dfB.head(8), use_container_width=True)

        idA_guess, descA_guess = guess_columns(dfA)
        idB_guess, descB_guess = guess_columns(dfB)

        c1, c2 = st.columns(2)
        with c1:
            idA = st.selectbox("File A — ID column", dfA.columns, index=list(dfA.columns).index(idA_guess),
                               help="Unique ID in File A (e.g., ID, Key).")
            descA = st.selectbox("File A — Description column", dfA.columns, index=list(dfA.columns).index(descA_guess),
                                 help="Text field to compare for File A.")
        with c2:
            idB = st.selectbox("File B — ID column", dfB.columns, index=list(dfB.columns).index(idB_guess),
                               help="Unique ID in File B.")
            descB = st.selectbox("File B — Description column", dfB.columns, index=list(dfB.columns).index(descB_guess),
                                 help="Text field to compare for File B.")

        # Clean + validate
        dfA = dfA.copy(); dfB = dfB.copy()
        dfA[idA] = dfA[idA].astype(str); dfB[idB] = dfB[idB].astype(str)
        dfA[descA] = dfA[descA].fillna("").astype(str)
        dfB[descB] = dfB[descB].fillna("").astype(str)

        badA = dfA[descA].str.strip().eq("").all()
        badB = dfB[descB].str.strip().eq("").all()
        if badA or badB:
            st.warning("Descriptions are empty in one or both files after cleaning. Please confirm your description columns.")
        else:
            if st.button("Compute similarities (A vs B)"):
                with st.spinner("Vectorizing and computing…"):
                    vec = build_vectorizer(ngram_min, ngram_max, min_df, max_df)
                    # Fit once on combined corpus so feature space aligns
                    combined = pd.concat([dfA[descA], dfB[descB]], ignore_index=True)
                    vec.fit(combined.tolist())
                    XA = vec.transform(dfA[descA].tolist())
                    XB = vec.transform(dfB[descB].tolist())

                    sim = cosine_similarity(XA, XB)
                    pairs = cross_long(sim, ids_a=dfA[idA].tolist(), ids_b=dfB[idB].tolist())

                    # Map descriptions from respective files
                    mapA = dict(zip(dfA[idA], dfA[descA]))
                    mapB = dict(zip(dfB[idB], dfB[descB]))
                    pairs["Desc_A"] = pairs["ID_A"].map(mapA)
                    pairs["Desc_B"] = pairs["ID_B"].map(mapB)

                    # Reorder + filter
                    pairs = pairs[["ID_A","Desc_A","ID_B","Desc_B","similarity"]]
                    if threshold > 0:
                        pairs = pairs[pairs["similarity"] >= threshold]

                    if topk_per_A > 0:
                        pairs = (pairs.sort_values("similarity", ascending=not sort_desc)
                                       .groupby("ID_A", as_index=False).head(topk_per_A))
                    elif sort_desc:
                        pairs = pairs.sort_values("similarity", ascending=False)

                st.success(f"Done. {len(pairs):,} A×B pairs.")
                st.dataframe(pairs.head(show_preview_rows), use_container_width=True)

                c1, c2 = st.columns(2)
                with c1:
                    st.download_button(
                        "⬇️ Download CSV",
                        data=pairs.to_csv(index=False).encode("utf-8"),
                        file_name="similarity_pairs_A_vs_B.csv",
                        mime="text/csv"
                    )
                with c2:
                    st.download_button(
                        "⬇️ Download Excel",
                        data=downloadable_excel(pairs, sheet_name="pairs_A_vs_B"),
                        file_name="similarity_pairs_A_vs_B.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

# -------------------------
# Friendly footer / FAQ
# -------------------------
with st.expander("Need help? (FAQ)"):
    st.markdown(
        "- **What is TF-IDF?** A way to turn text into numeric vectors; it down-weights very common words.\n"
        "- **Similarity score:** Cosine similarity ∈ [0, 1]; higher means more similar.\n"
        "- **Threshold:** Hide pairs below a chosen similarity to focus on strong matches.\n"
        "- **Top-K per ID_A:** Keep only the K best matches for each source story.\n"
        "- **n-grams:** Include phrases (e.g., 2-grams like 'user login'). Leaving defaults is fine."
    )

