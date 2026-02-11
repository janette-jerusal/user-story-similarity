# app.py — scalable Streamlit similarity (NO NxN matrices)
# Arrow-safe + Duplicate-column-safe + Topic/Status + Excel export

import io
import json
import traceback
from datetime import datetime, timezone
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# =========================================================
# Streamlit Setup
# =========================================================
st.set_page_config(page_title="User Story Similarity", layout="wide")

BUILD_TS = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

st.title("USCAP")
st.subheader("User Story Comparison Analysis Program")
st.caption(f"Build: {BUILD_TS} UTC • Scalable Top-K (no NxN matrices)")


# =========================================================
# Utilities
# =========================================================

def ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee unique column names (prevents Arrow duplicate crash)."""
    df = df.copy()
    seen = {}
    new_cols = []
    for c in df.columns:
        c = str(c).strip()
        if c not in seen:
            seen[c] = 1
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}__{seen[c]}")
    df.columns = new_cols
    return df


def make_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make DataFrame safe for st.dataframe / PyArrow.
    - Ensures unique columns
    - Removes timezone info
    - Converts complex objects to strings
    """
    out = ensure_unique_columns(df)

    # Remove timezone from datetime columns
    for col in out.columns:
        if pd.api.types.is_datetime64tz_dtype(out[col]):
            out[col] = out[col].dt.tz_convert(None)

    def safe_cell(x):
        if isinstance(x, (list, dict, set, tuple)):
            try:
                return json.dumps(x, default=str)
            except Exception:
                return str(x)
        return x

    for col in out.columns:
        if out[col].dtype == "object":
            col2 = out[col].map(safe_cell)
            if len(col2.dropna().map(type).unique()) > 1:
                col2 = col2.astype("string")
            out[col] = col2

    return out


def to_excel_bytes(df: pd.DataFrame, sheet_name="results") -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    buffer.seek(0)
    return buffer.read()


@st.cache_data(show_spinner=False)
def read_table(name: str, data: bytes) -> pd.DataFrame:
    bio = io.BytesIO(data)
    name = name.lower()

    if name.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(bio)
        except Exception:
            bio.seek(0)

    try:
        return pd.read_csv(bio)
    except Exception:
        bio.seek(0)
        return pd.read_excel(bio)


def clean_df(df: pd.DataFrame, id_col: str, desc_col: str) -> pd.DataFrame:
    df = df.copy()
    df[id_col] = df[id_col].astype(str).str.strip()
    df[desc_col] = (
        df[desc_col]
        .astype(str)
        .fillna("")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df = df[(df[id_col] != "") & (df[desc_col] != "")]
    df = df.drop_duplicates(subset=[id_col], keep="first")
    return df.reset_index(drop=True)


def build_vectorizer(ngram_min, ngram_max, min_df, max_df):
    if ngram_max < ngram_min:
        ngram_max = ngram_min
    return TfidfVectorizer(
        stop_words="english",
        ngram_range=(int(ngram_min), int(ngram_max)),
        min_df=int(min_df),
        max_df=float(max_df),
    )


# =========================================================
# Scalable Similarity (Top-K only)
# =========================================================

def topk_within(ids, texts, vectorizer, topk, threshold):
    X = vectorizer.fit_transform(texts)
    n = X.shape[0]
    k = min(topk + 1, n)

    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=k)
    nn.fit(X)

    distances, indices = nn.kneighbors(X)
    sims = 1 - distances

    rows = []
    seen = set()

    for i in range(n):
        id_a = ids[i]
        for jpos in range(k):
            j = indices[i, jpos]
            if j == i:
                continue
            sim = float(sims[i, jpos])
            if sim < threshold:
                continue
            id_b = ids[j]
            a, b = sorted([id_a, id_b])
            if (a, b) not in seen:
                seen.add((a, b))
                rows.append((a, b, sim))

    return pd.DataFrame(rows, columns=["ID_A", "ID_B", "similarity"])


def topk_A_vs_B(ids_a, texts_a, ids_b, texts_b, vectorizer, topk, threshold):
    vectorizer.fit(texts_a + texts_b)
    XA = vectorizer.transform(texts_a)
    XB = vectorizer.transform(texts_b)

    k = min(topk, XB.shape[0])

    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=k)
    nn.fit(XB)

    distances, indices = nn.kneighbors(XA)
    sims = 1 - distances

    rows = []
    for i in range(XA.shape[0]):
        id_a = ids_a[i]
        for jpos in range(k):
            j = indices[i, jpos]
            sim = float(sims[i, jpos])
            if sim >= threshold:
                rows.append((id_a, ids_b[j], sim))

    return pd.DataFrame(rows, columns=["ID_A", "ID_B", "similarity"])


# =========================================================
# UI
# =========================================================

mode = st.radio(
    "Comparison mode",
    ["One file (Top-K neighbors)", "Two files (A vs B Top-K)"],
    horizontal=True,
)

with st.sidebar:
    st.header("Compute Settings")
    threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.30, 0.01)
    topk = st.number_input("Top-K per ID_A", min_value=1, value=10)
    preview_rows = st.number_input("Preview rows", min_value=10, max_value=200, value=50)
    sort_desc = st.checkbox("Sort descending", value=True)

    st.divider()
    st.subheader("Vectorizer Settings")
    ngram_min = st.number_input("Min n-gram", 1, 5, 1)
    ngram_max = st.number_input("Max n-gram", 1, 5, 2)
    min_df = st.number_input("min_df", 1, value=1)
    max_df = st.slider("max_df", 0.1, 1.0, 1.0, 0.05)


# =========================================================
# ONE FILE MODE
# =========================================================

if mode.startswith("One"):
    file = st.file_uploader("Upload file (Excel/CSV)", type=["xlsx", "xls", "csv"])

    if file:
        df_raw = ensure_unique_columns(read_table(file.name, file.getvalue()))

        st.write("Preview")
        st.dataframe(make_arrow_safe(df_raw.head(10)), use_container_width=True)

        id_col = st.selectbox("ID column", df_raw.columns)
        desc_col = st.selectbox("Description column", df_raw.columns)

        df = clean_df(df_raw, id_col, desc_col)
        st.write(f"Usable rows: {len(df):,}")

        if st.button("Compute Similarities"):
            vec = build_vectorizer(ngram_min, ngram_max, min_df, max_df)
            pairs = topk_within(
                df[id_col].tolist(),
                df[desc_col].tolist(),
                vec,
                int(topk),
                float(threshold),
            )

            id_to_desc = dict(zip(df[id_col], df[desc_col]))
            pairs["Desc_A"] = pairs["ID_A"].map(id_to_desc)
            pairs["Desc_B"] = pairs["ID_B"].map(id_to_desc)

            if sort_desc:
                pairs = pairs.sort_values("similarity", ascending=False)

            st.success(f"Returned {len(pairs):,} pairs")
            st.dataframe(make_arrow_safe(pairs.head(int(preview_rows))), use_container_width=True)

            st.download_button(
                "Download CSV",
                pairs.to_csv(index=False).encode(),
                "similarity_onefile.csv",
            )

            st.download_button(
                "Download Excel",
                to_excel_bytes(pairs, "TopK_OneFile"),
                "similarity_onefile.xlsx",
            )


# =========================================================
# TWO FILE MODE
# =========================================================

else:
    fileA = st.file_uploader("Upload File A", type=["xlsx", "xls", "csv"], key="A")
    fileB = st.file_uploader("Upload File B", type=["xlsx", "xls", "csv"], key="B")

    if fileA and fileB:
        dfA = ensure_unique_columns(read_table(fileA.name, fileA.getvalue()))
        dfB = ensure_unique_columns(read_table(fileB.name, fileB.getvalue()))

        st.write("File A Preview")
        st.dataframe(make_arrow_safe(dfA.head()), use_container_width=True)

        st.write("File B Preview")
        st.dataframe(make_arrow_safe(dfB.head()), use_container_width=True)

        idA = st.selectbox("A: ID column", dfA.columns)
        descA = st.selectbox("A: Description column", dfA.columns)

        idB = st.selectbox("B: ID column", dfB.columns)
        descB = st.selectbox("B: Description column", dfB.columns)

        dfA = clean_df(dfA, idA, descA)
        dfB = clean_df(dfB, idB, descB)

        if st.button("Compute A vs B"):
            vec = build_vectorizer(ngram_min, ngram_max, min_df, max_df)

            pairs = topk_A_vs_B(
                dfA[idA].tolist(),
                dfA[descA].tolist(),
                dfB[idB].tolist(),
                dfB[descB].tolist(),
                vec,
                int(topk),
                float(threshold),
            )

            mapA = dict(zip(dfA[idA], dfA[descA]))
            mapB = dict(zip(dfB[idB], dfB[descB]))

            pairs["Desc_A"] = pairs["ID_A"].map(mapA)
            pairs["Desc_B"] = pairs["ID_B"].map(mapB)

            if sort_desc:
                pairs = pairs.sort_values("similarity", ascending=False)

            st.success(f"Returned {len(pairs):,} matches")
            st.dataframe(make_arrow_safe(pairs.head(int(preview_rows))), use_container_width=True)

            st.download_button(
                "Download CSV",
                pairs.to_csv(index=False).encode(),
                "similarity_A_vs_B.csv",
            )

            st.download_button(
                "Download Excel",
                to_excel_bytes(pairs, "TopK_A_vs_B"),
                "similarity_A_vs_B.xlsx",
            )

