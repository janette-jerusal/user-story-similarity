# app.py — scalable Streamlit similarity (NO NxN matrices) + Topic/Status + Excel export
import io
import traceback
from datetime import datetime, timezone
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# -------------------------
# Streamlit setup
# -------------------------
st.set_page_config(page_title="User Story Similarity", layout="wide")
BUILD_TS = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

st.title("USCAP")
st.subheader("User Story Comparison Analysis Program")
st.caption(f"Build: {BUILD_TS} UTC • Scalable Top-K (no NxN)")

# -------------------------
# Excel export helper
# -------------------------
def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "results") -> bytes:
    """
    Convert DataFrame to Excel bytes for Streamlit download.
    Uses openpyxl (in requirements).
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31])  # Excel sheet name max length = 31
    buffer.seek(0)
    return buffer.read()

# -------------------------
# File loading
# -------------------------
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
        bio.seek(0)
        return pd.read_excel(bio)

def guess_columns(df: pd.DataFrame) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Returns (id_col, desc_col, topic_col_or_None, status_col_or_None)
    """
    cols = [str(c).strip() for c in df.columns]
    norm = {c: c.lower().strip() for c in cols}

    id_aliases = {"id", "story id", "user story id", "issue id", "key", "ticket", "task id"}
    desc_aliases = {"description", "user story", "story", "summary", "title", "details", "acceptance criteria"}
    topic_aliases = {"topic", "category", "area", "module", "feature", "epic"}
    status_aliases = {"status", "state", "workflow state", "stage"}

    id_candidates = [c for c in cols if norm[c] in id_aliases]
    desc_candidates = [c for c in cols if norm[c] in desc_aliases]
    topic_candidates = [c for c in cols if norm[c] in topic_aliases]
    status_candidates = [c for c in cols if norm[c] in status_aliases]

    id_col = id_candidates[0] if id_candidates else cols[0]
    desc_col = desc_candidates[0] if desc_candidates else (cols[1] if len(cols) > 1 else cols[0])

    topic_col = topic_candidates[0] if topic_candidates else None
    status_col = status_candidates[0] if status_candidates else None

    return id_col, desc_col, topic_col, status_col

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

# -------------------------
# Core scalable similarity
# -------------------------
def topk_within_one_file(
    ids: list[str],
    texts: list[str],
    vectorizer: TfidfVectorizer,
    topk: int,
    threshold: float,
) -> pd.DataFrame:
    """
    Scalable: compute Top-K nearest neighbors for each row (cosine).
    Avoids NxN matrix.
    Returns unique undirected pairs once (ID_A < ID_B).
    """
    X = vectorizer.fit_transform(texts)  # sparse [N, D]

    n = X.shape[0]
    k = min(topk + 1, n)  # +1 to include self which we skip

    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=k, n_jobs=-1)
    nn.fit(X)

    distances, indices = nn.kneighbors(X, return_distance=True)
    sims = 1.0 - distances

    rows = []
    seen = set()  # (min_id, max_id)
    for i in range(n):
        id_a = ids[i]
        for jpos in range(k):
            j = int(indices[i, jpos])
            if j == i:
                continue
            sim = float(sims[i, jpos])
            if sim < threshold:
                continue
            id_b = ids[j]
            a, b = (id_a, id_b) if id_a < id_b else (id_b, id_a)
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            rows.append((a, b, sim))

    return pd.DataFrame(rows, columns=["ID_A", "ID_B", "similarity"])

def topk_A_vs_B(
    ids_a: list[str],
    texts_a: list[str],
    ids_b: list[str],
    texts_b: list[str],
    vectorizer: TfidfVectorizer,
    topk: int,
    threshold: float,
) -> pd.DataFrame:
    """
    Scalable: fit TF-IDF on combined corpus, then index B and query A for Top-K.
    Avoids full A×B matrix.
    """
    vectorizer.fit(texts_a + texts_b)
    XA = vectorizer.transform(texts_a)
    XB = vectorizer.transform(texts_b)

    k = min(topk, XB.shape[0])
    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=k, n_jobs=-1)
    nn.fit(XB)

    distances, indices = nn.kneighbors(XA, return_distance=True)
    sims = 1.0 - distances

    rows = []
    for i in range(XA.shape[0]):
        id_a = ids_a[i]
        for jpos in range(k):
            j = int(indices[i, jpos])
            sim = float(sims[i, jpos])
            if sim < threshold:
                continue
            id_b = ids_b[j]
            rows.append((id_a, id_b, sim))

    return pd.DataFrame(rows, columns=["ID_A", "ID_B", "similarity"])

# -------------------------
# UI controls
# -------------------------
mode = st.radio(
    "Comparison mode",
    ["One file (Top-K neighbors)", "Two files (A vs B Top-K)"],
    horizontal=True,
)

with st.sidebar:
    st.header("Compute settings")
    threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.30, 0.01)
    topk = st.number_input(
        "Top-K matches per ID_A",
        min_value=1,
        value=10,
        step=1,
        help="Scales to huge datasets because we compute only Top-K neighbors (no NxN).",
    )
    sort_desc = st.checkbox("Sort by similarity (desc)", value=True)
    preview_rows = st.number_input("Preview rows", min_value=10, max_value=200, value=50, step=10)

with st.expander("Vectorizer settings (optional)", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        ngram_min = st.number_input("Min n-gram", 1, 5, 1, 1)
        ngram_max = st.number_input("Max n-gram", 1, 5, 2, 1)
    with c2:
        min_df = st.number_input("min_df", 1, value=1, step=1)
    with c3:
        max_df = st.slider("max_df", 0.1, 1.0, 1.0, 0.05)

# -------------------------
# ONE FILE
# -------------------------
if mode.startswith("One file"):
    file1 = st.file_uploader("Upload a file (Excel/CSV)", type=["xlsx", "xls", "csv"], key="one")
    if file1 is not None:
        df_raw = read_table_from_bytes(file1.name, file1.getvalue())
        st.write("**Preview**")
        st.dataframe(df_raw.head(10), width="stretch")

        id_guess, desc_guess, topic_guess, status_guess = guess_columns(df_raw)

        c1, c2 = st.columns(2)
        with c1:
            id_col = st.selectbox("ID column", df_raw.columns, index=list(df_raw.columns).index(id_guess))
        with c2:
            desc_col = st.selectbox("Description column", df_raw.columns, index=list(df_raw.columns).index(desc_guess))

        c3, c4 = st.columns(2)
        with c3:
            topic_options = ["(none)"] + list(df_raw.columns)
            topic_index = 0 if topic_guess is None else (topic_options.index(topic_guess) if topic_guess in topic_options else 0)
            topic_col = st.selectbox("Topic column (optional)", topic_options, index=topic_index)
        with c4:
            status_options = ["(none)"] + list(df_raw.columns)
            status_index = 0 if status_guess is None else (status_options.index(status_guess) if status_guess in status_options else 0)
            status_col = st.selectbox("Status column (optional)", status_options, index=status_index)

        df = clean_df(df_raw, id_col, desc_col)
        st.write(f"Usable rows after cleaning: **{len(df):,}**")

        if st.button("Compute similarities (Top-K)", key="go_one"):
            try:
                if len(df) < 2:
                    st.error("Need at least 2 rows.")
                    st.stop()

                ids = df[id_col].astype(str).tolist()
                texts = df[desc_col].astype(str).fillna("").tolist()

                vec = build_vectorizer(ngram_min, ngram_max, min_df, max_df)

                st.info("Computing Top-K neighbors (no NxN matrix)…")
                pairs = topk_within_one_file(ids, texts, vec, int(topk), float(threshold))

                # Attach descriptions + topic/status
                id_to_desc = dict(zip(ids, texts))

                topic_map = None
                status_map = None
                if topic_col != "(none)" and topic_col in df.columns:
                    topic_map = dict(zip(df[id_col].astype(str), df[topic_col].astype(str).fillna("").tolist()))
                if status_col != "(none)" and status_col in df.columns:
                    status_map = dict(zip(df[id_col].astype(str), df[status_col].astype(str).fillna("").tolist()))

                pairs["Desc_A"] = pairs["ID_A"].map(id_to_desc)
                pairs["Desc_B"] = pairs["ID_B"].map(id_to_desc)

                pairs["Topic_A"] = pairs["ID_A"].map(topic_map) if topic_map else ""
                pairs["Status_A"] = pairs["ID_A"].map(status_map) if status_map else ""
                pairs["Topic_B"] = pairs["ID_B"].map(topic_map) if topic_map else ""
                pairs["Status_B"] = pairs["ID_B"].map(status_map) if status_map else ""

                pairs = pairs[
                    ["ID_A", "Topic_A", "Status_A", "Desc_A", "ID_B", "Topic_B", "Status_B", "Desc_B", "similarity"]
                ]

                if sort_desc:
                    pairs = pairs.sort_values("similarity", ascending=False)

                st.success(f"Done. Returned **{len(pairs):,}** unique pairs (deduped).")
                st.dataframe(pairs.head(int(preview_rows)), width="stretch")

                dl1, dl2 = st.columns(2)
                with dl1:
                    st.download_button(
                        "⬇️ Download CSV",
                        data=pairs.to_csv(index=False).encode("utf-8"),
                        file_name="similarity_topk_onefile.csv",
                        mime="text/csv",
                    )
                with dl2:
                    st.download_button(
                        "⬇️ Download Excel",
                        data=to_excel_bytes(pairs, sheet_name="TopK_OneFile"),
                        file_name="similarity_topk_onefile.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

            except Exception:
                st.error("Compute crashed. Traceback:")
                st.code(traceback.format_exc())
                st.stop()

# -------------------------
# TWO FILES
# -------------------------
else:
    fileA = st.file_uploader("Upload File A (Excel/CSV)", type=["xlsx", "xls", "csv"], key="A")
    fileB = st.file_uploader("Upload File B (Excel/CSV)", type=["xlsx", "xls", "csv"], key="B")

    if fileA is not None and fileB is not None:
        dfA_raw = read_table_from_bytes(fileA.name, fileA.getvalue())
        dfB_raw = read_table_from_bytes(fileB.name, fileB.getvalue())

        st.write("**File A Preview**")
        st.dataframe(dfA_raw.head(8), width="stretch")
        st.write("**File B Preview**")
        st.dataframe(dfB_raw.head(8), width="stretch")

        idA_guess, descA_guess, topicA_guess, statusA_guess = guess_columns(dfA_raw)
        idB_guess, descB_guess, topicB_guess, statusB_guess = guess_columns(dfB_raw)

        c1, c2 = st.columns(2)
        with c1:
            idA = st.selectbox("A: ID column", dfA_raw.columns, index=list(dfA_raw.columns).index(idA_guess))
            descA = st.selectbox("A: Description column", dfA_raw.columns, index=list(dfA_raw.columns).index(descA_guess))
        with c2:
            idB = st.selectbox("B: ID column", dfB_raw.columns, index=list(dfB_raw.columns).index(idB_guess))
            descB = st.selectbox("B: Description column", dfB_raw.columns, index=list(dfB_raw.columns).index(descB_guess))

        c3, c4 = st.columns(2)
        with c3:
            topicA_opts = ["(none)"] + list(dfA_raw.columns)
            statusA_opts = ["(none)"] + list(dfA_raw.columns)
            topicA_idx = 0 if topicA_guess is None else (topicA_opts.index(topicA_guess) if topicA_guess in topicA_opts else 0)
            statusA_idx = 0 if statusA_guess is None else (statusA_opts.index(statusA_guess) if statusA_guess in statusA_opts else 0)
            topicA = st.selectbox("A: Topic column (optional)", topicA_opts, index=topicA_idx)
            statusA = st.selectbox("A: Status column (optional)", statusA_opts, index=statusA_idx)
        with c4:
            topicB_opts = ["(none)"] + list(dfB_raw.columns)
            statusB_opts = ["(none)"] + list(dfB_raw.columns)
            topicB_idx = 0 if topicB_guess is None else (topicB_opts.index(topicB_guess) if topicB_guess in topicB_opts else 0)
            statusB_idx = 0 if statusB_guess is None else (statusB_opts.index(statusB_guess) if statusB_guess in statusB_opts else 0)
            topicB = st.selectbox("B: Topic column (optional)", topicB_opts, index=topicB_idx)
            statusB = st.selectbox("B: Status column (optional)", statusB_opts, index=statusB_idx)

        dfA = clean_df(dfA_raw, idA, descA)
        dfB = clean_df(dfB_raw, idB, descB)
        st.write(f"Usable A rows: **{len(dfA):,}** • Usable B rows: **{len(dfB):,}**")

        if st.button("Compute A vs B (Top-K)", key="go_two"):
            try:
                if dfA.empty or dfB.empty:
                    st.error("No usable rows in A or B after cleaning.")
                    st.stop()

                ids_a = dfA[idA].astype(str).tolist()
                texts_a = dfA[descA].astype(str).fillna("").tolist()
                ids_b = dfB[idB].astype(str).tolist()
                texts_b = dfB[descB].astype(str).fillna("").tolist()

                vec = build_vectorizer(ngram_min, ngram_max, min_df, max_df)

                st.info("Indexing B and querying Top-K for each A (no A×B matrix)…")
                pairs = topk_A_vs_B(ids_a, texts_a, ids_b, texts_b, vec, int(topk), float(threshold))

                # Attach descriptions
                mapA_desc = dict(zip(ids_a, texts_a))
                mapB_desc = dict(zip(ids_b, texts_b))

                # Attach topic/status (optional)
                mapA_topic = None
                mapA_status = None
                mapB_topic = None
                mapB_status = None

                if topicA != "(none)" and topicA in dfA.columns:
                    mapA_topic = dict(zip(dfA[idA].astype(str), dfA[topicA].astype(str).fillna("").tolist()))
                if statusA != "(none)" and statusA in dfA.columns:
                    mapA_status = dict(zip(dfA[idA].astype(str), dfA[statusA].astype(str).fillna("").tolist()))
                if topicB != "(none)" and topicB in dfB.columns:
                    mapB_topic = dict(zip(dfB[idB].astype(str), dfB[topicB].astype(str).fillna("").tolist()))
                if statusB != "(none)" and statusB in dfB.columns:
                    mapB_status = dict(zip(dfB[idB].astype(str), dfB[statusB].astype(str).fillna("").tolist()))

                pairs["Desc_A"] = pairs["ID_A"].map(mapA_desc)
                pairs["Desc_B"] = pairs["ID_B"].map(mapB_desc)

                pairs["Topic_A"] = pairs["ID_A"].map(mapA_topic) if mapA_topic else ""
                pairs["Status_A"] = pairs["ID_A"].map(mapA_status) if mapA_status else ""
                pairs["Topic_B"] = pairs["ID_B"].map(mapB_topic) if mapB_topic else ""
                pairs["Status_B"] = pairs["ID_B"].map(mapB_status) if mapB_status else ""

                pairs = pairs[
                    ["ID_A", "Topic_A", "Status_A", "Desc_A", "ID_B", "Topic_B", "Status_B", "Desc_B", "similarity"]
                ]

                if sort_desc:
                    pairs = pairs.sort_values("similarity", ascending=False)

                st.success(f"Done. Returned **{len(pairs):,}** A→B matches (Top-K).")
                st.dataframe(pairs.head(int(preview_rows)), width="stretch")

                dl1, dl2 = st.columns(2)
                with dl1:
                    st.download_button(
                        "⬇️ Download CSV",
                        data=pairs.to_csv(index=False).encode("utf-8"),
                        file_name="similarity_topk_A_vs_B.csv",
                        mime="text/csv",
                    )
                with dl2:
                    st.download_button(
                        "⬇️ Download Excel",
                        data=to_excel_bytes(pairs, sheet_name="TopK_A_vs_B"),
                        file_name="similarity_topk_A_vs_B.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

            except Exception:
                st.error("Compute crashed. Traceback:")
                st.code(traceback.format_exc())
                st.stop()

# -------------------------
# FAQ
# -------------------------
with st.expander("FAQ"):
    st.markdown(
        "- This version is **scalable** because it never builds an NxN or A×B similarity matrix.\n"
        "- Increase **Top-K** for more matches per story.\n"
        "- Increase **threshold** to reduce results.\n"
        "- For huge files, start with Top-K=5–20."
    )
