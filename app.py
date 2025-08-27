# app.py
# User Story Similarity — Raytheon-styled, sleek & professional
# - One-file mode: compares each pair ONCE (upper triangle, no self-pairs)
# - Two-file mode: A vs B only (rectangular)
# - Outputs: ID_A, Desc_A, ID_B, Desc_B, similarity

import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Page config & Styling
# -------------------------
st.set_page_config(page_title="User Story Similarity", layout="wide")

# Raytheon palette CSS — sleek, high-tech look
st.markdown("""
<style>
:root {
  --ray-red: #cc0000;
  --ray-dark: #1c1c1c;
  --ray-gray: #6e6e6e;
  --ray-light: #f5f5f5;
  --ring: rgba(204,0,0,0.35);
}
html, body, [class*="css"]  {
  font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  color: var(--ray-dark);
  background-color: var(--ray-light);
}
h1,h2,h3 { letter-spacing: -0.02em; }

/* Header band */
.app-header {
  border: 1px solid rgba(204,0,0,0.4);
  border-radius: 14px;
  padding: 20px;
  background: linear-gradient(90deg, var(--ray-red) 0%, #7a0000 100%);
  color: white;
  box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.app-header h1 { margin: 0; font-weight: 700; letter-spacing: -0.5px; }
.app-header-sub { font-size: 0.9rem; font-weight: 500; color: #f5f5f5; }

/* Callouts */
.callout {
  border-left: 4px solid var(--ray-red);
  border-radius: 8px;
  padding: 12px 14px;
  background: rgba(204,0,0,0.05);
  color: var(--ray-dark);
  margin-top: 6px;
}

/* KPI chips */
.kpi {
  border-radius: 10px;
  padding: 12px 14px;
  background: var(--ray-dark);
  color: white;
  font-weight: 600;
  text-align: center;
  box-shadow: 0 2px 6px rgba(0,0,0,0.25);
}

/* Buttons */
.stDownloadButton button, .stButton button {
  border-radius: 8px !important;
  padding: 0.6rem 1rem !important;
  border: none;
  font-weight: 600;
  background: var(--ray-red);
  color: white;
  box-shadow: 0 2px 6px rgba(0,0,0,0.25);
}
.stDownloadButton button:hover, .stButton button:hover {
  background: #a30000 !important;
  box-shadow: 0 0 0 3px var(--ring);
}

/* Inputs */
input, textarea, select { border-radius: 6px !important; }
input:focus, textarea:focus, select:focus {
  box-shadow: 0 0 0 3px var(--ring);
  border-color: var(--ray-red) !important;
}

/* Section titles */
.section-title { font-weight: 700; color: var(--ray-red); margin-top: 1rem; margin-bottom: 0.25rem; }

/* Dataframe tweaks */
.dataframe th { background-color: var(--ray-dark) !important; color: white !important; }

/* Container spacing */
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

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
    return TfidfVectorizer(stop_words="english", ngram_range=(ngram_min, ngram_max), min_df=min_df, max_df=max_df)

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
# Header
# -------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>🔎 User Story Similarity</h1>
        <div class="app-header-sub">Raytheon-styled • TF-IDF • Cosine similarity • One-to-one pairs</div>
    </div>
    """,
    unsafe_allow_html=True
)
st.caption(f"Build: {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC")

# -------------------------
# Controls
# -------------------------
col_mode, col_note = st.columns([0.55, 0.45])
with col_mode:
    mode = st.radio(
        "Comparison mode",
        ["One file (all vs all, once each)", "Two files (A vs B)"],
        horizontal=True,
        help="Compare within a single file (each pair once) or across two files (A vs B).",
    )
with col_note:
    st.markdown(
        """
        <div class="callout">
        <b>How results are built</b><br/>
        • <i>One file:</i> upper triangle (no A↔B duplicates, no self-pairs).<br/>
        • <i>Two files:</i> A vs B once by definition (no B vs A).
        </div>
        """,
        unsafe_allow_html=True
    )

with st.sidebar:
    st.subheader("Filters")
    threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.35, 0.01)
    topk_per_A = st.number_input("Top-K per ID_A (0 = no cap)", min_value=0, value=0, step=1)
    sort_desc = st.checkbox("Sort by similarity (high → low)", value=True)
    show_preview_rows = st.number_input("Preview rows", min_value=10, max_value=200, value=50, step=10)

with st.expander("Advanced settings", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        ngram_min = st.number_input("Min n-gram", 1, 3, 1, 1)
        ngram_max = st.number_input("Max n-gram", 1, 3, 2, 1)
    with c2:
        min_df = st.number_input("min_df", 1, value=1, step=1)
    with c3:
        max_df = st.slider("max_df", 0.1, 1.0, 1.0, 0.05)

# -------------------------
# ONE FILE MODE
# -------------------------
if mode.startswith("One file"):
    file1 = st.file_uploader("Upload a file (Excel/CSV)", type=["xlsx","xls","csv"])
    if file1 is not None:
        df = read_table(file1)
        id_guess, desc_guess = guess_columns(df)

        c1, c2 = st.columns(2)
        with c1:
            id_col = st.selectbox("ID column", df.columns, index=list(df.columns).index(id_guess))
        with c2:
            desc_col = st.selectbox("Description column", df.columns, index=list(df.columns).index(desc_guess))

        st.dataframe(df[[id_col, desc_col]].head(10), use_container_width=True)

        df = df.copy()
        df[id_col] = df[id_col].astype(str)
        df[desc_col] = df[desc_col].fillna("").astype(str)

        if st.button("▶️ Compute similarities (One file)"):
            vec = build_vectorizer(ngram_min, ngram_max, min_df, max_df)
            X = vec.fit_transform(df[desc_col].tolist())
            sim = cosine_similarity(X, X)
            pairs = upper_triangle_long(sim, ids=df[id_col].tolist())

            # Map descriptions
            id_to_desc = dict(zip(df[id_col], df[desc_col]))
            pairs["Desc_A"] = pairs["ID_A"].map(id_to_desc)
            pairs["Desc_B"] = pairs["ID_B"].map(id_to_desc)
            pairs = pairs[["ID_A","Desc_A","ID_B","Desc_B","similarity"]]

            if threshold > 0:
                pairs = pairs[pairs["similarity"] >= threshold]
            if topk_per_A > 0:
                pairs = (pairs.sort_values("similarity", ascending=not sort_desc)
                               .groupby("ID_A", as_index=False).head(topk_per_A))
            elif sort_desc:
                pairs = pairs.sort_values("similarity", ascending=False)

            st.dataframe(pairs.head(show_preview_rows), use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.download_button("⬇️ Download CSV",
                                   data=pairs.to_csv(index=False).encode("utf-8"),
                                   file_name="similarity_pairs_once.csv",
                                   mime="text/csv")
            with c2:
                st.download_button("⬇️ Download Excel",
                                   data=downloadable_excel(pairs, "pairs_once"),
                                   file_name="similarity_pairs_once.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -------------------------
# TWO FILES MODE
# -------------------------
else:
    fileA = st.file_uploader("Upload File A", type=["xlsx","xls","csv"], key="A")
    fileB = st.file_uploader("Upload File B", type=["xlsx","xls","csv"], key="B")

    if fileA is not None and fileB is not None:
        dfA = read_table(fileA)
        dfB = read_table(fileB)
        idA_guess, descA_guess = guess_columns(dfA)
        idB_guess, descB_guess = guess_columns(dfB)

        c1, c2 = st.columns(2)
        with c1:
            idA = st.selectbox("File A — ID", dfA.columns, index=list(dfA.columns).index(idA_guess))
            descA = st.selectbox("File A — Description", dfA.columns, index=list(dfA.columns).index(descA_guess))
        with c2:
            idB = st.selectbox("File B — ID", dfB.columns, index=list(dfB.columns).index(idB_guess))
            descB = st.selectbox("File B — Description", dfB.columns, index=list(dfB.columns).index(descB_guess))

        st.dataframe(dfA[[idA, descA]].head(5), use_container_width=True)
        st.dataframe(dfB[[idB, descB]].head(5), use_container_width=True)

        dfA = dfA.copy(); dfB = dfB.copy()
        dfA[idA] = dfA[idA].astype(str); dfB[idB] = dfB[idB].astype(str)
        dfA[descA] = dfA[descA].fillna("").astype(str)
        dfB[descB] = dfB[descB].fillna("").astype(str)

        if st.button("▶️ Compute similarities (A vs B)"):
            vec = build_vectorizer(ngram_min, ngram_max, min_df, max_df)
            combined = pd.concat([dfA[descA], dfB[descB]], ignore_index=True)
            vec.fit(combined.tolist())
            XA = vec.transform(dfA[descA].tolist())
            XB = vec.transform(dfB[descB].tolist())

            sim = cosine_similarity(XA, XB)
            pairs = cross_long(sim, ids_a=dfA[idA].tolist(), ids_b=dfB[idB].tolist())

            mapA = dict(zip(dfA[idA], dfA[descA]))
            mapB = dict(zip(dfB[idB], dfB[descB]))
            pairs["Desc_A"] = pairs["ID_A"].map(mapA)
            pairs["Desc_B"] = pairs["ID_B"].map(mapB)
            pairs = pairs[["ID_A","Desc_A","ID_B","Desc_B","similarity"]]

            if threshold > 0:
                pairs = pairs[pairs["similarity"] >= threshold]
            if topk_per_A > 0:
                pairs = (pairs.sort_values("similarity", ascending=not sort_desc)
                               .groupby("ID_A", as_index=False).head(topk_per_A))
            elif sort_desc:
                pairs = pairs.sort_values("similarity", ascending=False)

            st.dataframe(pairs.head(show_preview_rows), use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.download_button("⬇️ Download CSV",
                                   data=pairs.to_csv(index=False).encode("utf-8"),
                                   file_name="similarity_pairs_A_vs_B.csv",
                                   mime="text/csv")
            with c2:
                st.download_button("⬇️ Download Excel",
                                   data=downloadable_excel(pairs, "pairs_A_vs_B"),
                                   file_name="similarity_pairs_A_vs_B.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -------------------------
# Footer FAQ
# -------------------------
with st.expander("Need help? (FAQ)"):
    st.markdown(
        "- **What is TF-IDF?** Turns text into weighted vectors, down-weighting common words.\n"
        "- **Similarity score:** Cosine similarity [0,1]; higher = more similar.\n"
        "- **Threshold:** Hide weak matches.\n"
        "- **Top-K:** Keep only top K matches per story.\n"
        "- **n-grams:** Include short phrases (defaults are fine)."
    )

