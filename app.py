"""
app.py  – User‑story similarity Streamlit app
--------------------------------------------
Only two edits were made compared with your repo:
  • CLEANING df1 / df2 to remove/replace NaNs
  • A tiny tweak to the similarity threshold slider default (=0.6)
Everything else is identical.
"""

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------- core similarity logic ----------
def compute_similarity(df1: pd.DataFrame,
                       df2: pd.DataFrame,
                       threshold: float = 0.6) -> pd.DataFrame:
    """
    Compare user‑stories in df1 vs df2 (columns: 'id', 'desc').
    Returns rows where cosine similarity ≥ threshold.
    """
    # --- NEW: robust cleaning to avoid NaNs going into TfidfVectorizer ----
    needed_cols = ["id", "desc"]

    df1 = df1[needed_cols].copy()
    df2 = df2[needed_cols].copy()

    for df in (df1, df2):
        df["desc"] = (df["desc"]
                      .astype(str)          # force to string
                      .fillna("")           # replace NaN with empty string
                      .str.strip())         # strip whitespace

    # if all descriptions are empty, bail early
    if (df1["desc"].eq("").all() or df2["desc"].eq("").all()):
        st.error("One of the files has no usable 'desc' text.")
        return pd.DataFrame(columns=["id_1", "id_2", "similarity"])

    # ---------------------------------------------------------------------
    combined = pd.concat([df1["desc"], df2["desc"]], ignore_index=True)

    tfidf = TfidfVectorizer().fit_transform(combined)
    n1 = len(df1)
    tfidf_1 = tfidf[:n1]
    tfidf_2 = tfidf[n1:]

    sim_matrix = cosine_similarity(tfidf_1, tfidf_2)

    # build result dataframe
    pairs = []
    for i, id1 in enumerate(df1["id"]):
        for j, id2 in enumerate(df2["id"]):
            sim = sim_matrix[i, j]
            if sim >= threshold:
                pairs.append({"id_1": id1,
                              "id_2": id2,
                              "similarity": round(float(sim), 4)})

    return pd.DataFrame(pairs)


# ---------- Streamlit UI ----------
st.title("📊 User‑Story Similarity Comparator")

uploaded_1 = st.file_uploader("Upload file 1 (CSV / Excel)", type=["csv", "xls", "xlsx"])
uploaded_2 = st.file_uploader("Upload file 2 (CSV / Excel)", type=["csv", "xls", "xlsx"])

threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.60, 0.01)

def load_any(f):
    if f.name.endswith((".xls", ".xlsx")):
        return pd.read_excel(f)
    return pd.read_csv(f)

if uploaded_1 and uploaded_2:
    try:
        df1 = load_any(uploaded_1)
        df2 = load_any(uploaded_2)

        if not {"id", "desc"}.issubset(df1.columns) \
           or not {"id", "desc"}.issubset(df2.columns):
            st.error("Both files must contain **id** and **desc** columns.")
            st.stop()

        st.success("Files successfully loaded!")
        if st.button("🔍 Compare"):
            with st.spinner("Computing similarities …"):
                result_df = compute_similarity(df1, df2, threshold)

            if result_df.empty:
                st.info("No pairs met the threshold.")
            else:
                st.subheader(f"Matches (≥ {threshold})")
                st.dataframe(result_df)

    except Exception as e:
        st.error(f"❌ Error reading files: {e}")
else:
    st.info("Please upload two files to begin.")
