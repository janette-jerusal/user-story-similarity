# app.py
# User Story Similarity — robust UI + safe exports

import io
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------
# Streamlit setup
# -------------------------
st.set_page_config(page_title="User Story Similarity", layout="wide")
BUILD_TS = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
st.title("USCAP")
st.caption(f"Build: {BUILD_TS} UTC • One-to-one pairs only")

# -------------------------
# Helpers (cached)
# -------------------------
@st.cache_data(show_spinner=False)
def _read_bytes(uploaded_bytes: bytes) -> bytes:
    # Separate cache primitive so pandas engine detection reuses this content
    return uploaded_bytes

@st.cache_data(show_spinner=False)
def read_table_from_bytes(name: str, uploaded_bytes: bytes) -> pd.DataFrame:
    """
    Robust loader: try Excel (openpyxl/xlrd) then CSV; falls back gracefully.
    Cached by file bytes to avoid re-parsing on rerun.
    """
    data = _read_bytes(uploaded_bytes)
    bio = io.BytesIO(data)

    lower = name.lower()
    if lower.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(bio)
        except Exception:
            bio.seek(0)

    try:
        bio.seek(0)
        return pd.read_csv(bio)
    except Exception:
        # Final attempt: Excel without trusting extension
        bio.seek(0)
        return pd.read_excel(bio)

def guess_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cols = [str(c).strip() for c in df.columns]
    norm = {c: c.lower().strip() for c in cols}

    id_aliases = {"id", "story id", "user story id", "issue id", "key", "ticket", "task id"}
    desc_aliases = {"description", "user story", "story", "summary", "title", "details", "acceptance criteria"}

    id_candidates = [c for c in cols if norm[c] in id_aliases]
    desc_candidates = [c for c in cols if norm[c] in desc_aliases]

    id_col = id_candidates[0] if id_candidates else cols[0]
    desc_col = (
        desc_candidates[0]
        if desc_candidates
        else (cols[1] if len(cols) > 1 and cols[1] != id_col else id_col)
    )
    return id_col, desc_col

def build_vectorizer(ngram_min=1, ngram_max=2, min_df=1, max_df=1.0) -> TfidfVectorizer:
    # Defensive: ensure min<=max
    if ngram_max < ngram_min:
        ngram_max = ngram_min
    return TfidfVectorizer(
        stop_words="english",
        ngram_range=(int(ngram_min), int(ngram_max)),
        min_df=int(min_df),
        max_df=float(max_df),
    )

def _upper_triangle_long(sim_matrix: np.ndarray, ids) -> pd.DataFrame:
    sim_df = pd.DataFrame(sim_matrix, index=ids, columns=ids)
    upper_only = sim_df.where(np.triu(np.ones(sim_df.shape, dtype=bool), k=1))
    return (
        upper_only.stack()
        .reset_index()
        .rename(columns={"level_0": "ID_A", "level_1": "ID_B", 0: "similarity"})
    )

def _cross_long(sim_matrix: np.ndarray, ids_a, ids_b) -> pd.DataFrame:
    sim_df = pd.DataFrame(sim_matrix, index=ids_a, columns=ids_b)
    return (
        sim_df.stack()
        .reset_index()
        .rename(columns={"level_0": "ID_A", "level_1": "ID_B", 0: "similarity"})
    )

def _clean_df(df: pd.DataFrame, id_col: str, desc_col: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df[id_col] = df[id_col].astype(str).str.strip()
    df[desc_col] = df[desc_col].astype(str).fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    # Drop rows with empty description or empty id
    df = df[(df[desc_col] != "") & (df[id_col] != "")]
    # Deduplicate IDs to keep first occurrence
    df = df.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)
    return df

def downloadable_excel(
    df_pairs: pd.DataFrame,
    sheet_name: str,
    meta: Optional[dict] = None,
) -> Optional[bytes]:
    """
    Export pairs + a small Settings sheet. Returns None if no engine installed.
    """
    buf = io.BytesIO()
    engine = None
    try:
        import xlsxwriter  # noqa: F401
        engine = "xlsxwriter"
    except Exception:
        try:
            import openpyxl  # noqa: F401
            engine = "openpyxl"
        except Exception:
            return None

    with pd.ExcelWriter(buf, engine=engine) as w:
        df_pairs.to_excel(w, index=False, sheet_name=sheet_name)
        if meta:
            pd.DataFrame(list(meta.items()), columns=["Setting", "Value"]).to_excel(
                w, index=False, sheet_name="Settings"
            )
    buf.seek(0)
    return buf.read()

# -------------------------
# Controls
# -------------------------
mode = st.radio(
    "Comparison mode",
    ["One file (all vs all, once each)", "Two files (A vs B)"],
    horizontal=True,
    key="mode",
    help="Compare within one file (each pair once) or File A vs File B.",
)

st.info(
    "Each pair is compared **once** only.\n"
    "• **One file**: uses the upper triangle → no A↔B duplicates, no self-pairs.\n"
    "• **Two files**: A vs B pairs only (no B vs A)."
)

with st.sidebar:
    st.header("Filters")
    threshold = st.slider(
        "Similarity threshold", 0.0, 1.0, 0.30, 0.01,
        help="Hide pairs below this cosine similarity."
    )
    topk_per_A = st.number_input(
        "Top-K per ID_A (0 = no limit)", min_value=0, value=0, step=1,
        help="Keep at most K best matches for each ID_A (after threshold)."
    )
    sort_desc = st.checkbox("Sort by similarity (desc)", value=True)
    show_preview_rows = st.number_input("Preview rows", min_value=10, max_value=200, value=50, step=10)

with st.expander("Advanced settings (optional)", expanded=False):
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
    file1 = st.file_uploader(
        "Upload a file (Excel/CSV) with **ID** and **Description** columns",
        type=["xlsx", "xls", "csv"],
        key="onefile",
    )

    if file1 is not None:
        df_raw = read_table_from_bytes(file1.name, file1.getvalue())
        st.write("**Data preview**")
        st.dataframe(df_raw.head(10), use_container_width=True)

        id_guess, desc_guess = guess_columns(df_raw)
        c1, c2 = st.columns(2)
        with c1:
            id_col = st.selectbox("ID column", df_raw.columns, index=list(df_raw.columns).index(id_guess), key="id1")
        with c2:
            desc_col = st.selectbox("Description column", df_raw.columns, index=list(df_raw.columns).index(desc_guess), key="desc1")

        df = _clean_df(df_raw, id_col, desc_col)
        if df.empty or df[desc_col].str.strip().eq("").all():
            st.warning("No usable rows after cleaning. Please confirm your columns.")
        else:
            if st.button("Compute similarities (One file)", key="go1"):
                with st.spinner("Vectorizing and computing…"):
                    vec = build_vectorizer(ngram_min, ngram_max, min_df, max_df)
                    X = vec.fit_transform(df[desc_col].tolist())
                    if X.shape[0] < 2:
                        st.warning("Need at least 2 stories to compare.")
                    else:
                        sim = cosine_similarity(X, X)
                        pairs = _upper_triangle_long(sim, ids=df[id_col].tolist())
                        # Map descriptions
                        id_to_desc = dict(zip(df[id_col], df[desc_col]))
                        pairs["Desc_A"] = pairs["ID_A"].map(id_to_desc)
                        pairs["Desc_B"] = pairs["ID_B"].map(id_to_desc)
                        # Reorder + filter
                        pairs = pairs[["ID_A", "Desc_A", "ID_B", "Desc_B", "similarity"]]
                        if threshold > 0:
                            pairs = pairs[pairs["similarity"] >= threshold]
                        if topk_per_A > 0:
                            pairs = (
                                pairs.sort_values("similarity", ascending=not sort_desc)
                                .groupby("ID_A", as_index=False, sort=False)
                                .head(topk_per_A)
                            )
                        elif sort_desc:
                            pairs = pairs.sort_values("similarity", ascending=False)

                        st.success(f"Done. {len(pairs):,} unique pairs (A→B only).")
                        st.dataframe(pairs.head(show_preview_rows), use_container_width=True)

                        meta = {
                            "Build UTC": BUILD_TS,
                            "Mode": "One file",
                            "IDs": id_col,
                            "Descriptions": desc_col,
                            "Threshold": threshold,
                            "TopK per A": topk_per_A,
                            "Sort desc": sort_desc,
                            "n-gram": f"{ngram_min}-{ngram_max}",
                            "min_df": min_df,
                            "max_df": max_df,
                        }

                        c1, c2 = st.columns(2)
                        with c1:
                            st.download_button(
                                "⬇️ Download CSV",
                                data=pairs.to_csv(index=False).encode("utf-8"),
                                file_name="similarity_pairs_once.csv",
                                mime="text/csv",
                            )
                        with c2:
                            xlsx_bytes = downloadable_excel(pairs, sheet_name="pairs_once", meta=meta)
                            if xlsx_bytes:
                                st.download_button(
                                    "⬇️ Download Excel",
                                    data=xlsx_bytes,
                                    file_name="similarity_pairs_once.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                )
                            else:
                                st.caption("ℹ️ Excel export unavailable (install `xlsxwriter` or `openpyxl`). CSV works.")

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
        st.dataframe(dfA_raw.head(8), use_container_width=True)
        st.write("**File B Preview**")
        st.dataframe(dfB_raw.head(8), use_container_width=True)

        idA_guess, descA_guess = guess_columns(dfA_raw)
        idB_guess, descB_guess = guess_columns(dfB_raw)

        c1, c2 = st.columns(2)
        with c1:
            idA = st.selectbox("File A — User Story ID Column", dfA_raw.columns, index=list(dfA_raw.columns).index(idA_guess), key="idA")
            descA = st.selectbox("File A — User Story Description column", dfA_raw.columns, index=list(dfA_raw.columns).index(descA_guess), key="descA")
        with c2:
            idB = st.selectbox("File B — User Story ID column", dfB_raw.columns, index=list(dfB_raw.columns).index(idB_guess), key="idB")
            descB = st.selectbox("File B — User Story Description column", dfB_raw.columns, index=list(dfB_raw.columns).index(descB_guess), key="descB")

        dfA = _clean_df(dfA_raw, idA, descA)
        dfB = _clean_df(dfB_raw, idB, descB)

        if dfA.empty or dfB.empty:
            st.warning("No usable rows after cleaning in one or both files.")
        else:
            if st.button("Compute Similarities (A vs B)", key="go2"):
                with st.spinner("Vectorizing and computing…"):
                    vec = build_vectorizer(ngram_min, ngram_max, min_df, max_df)
                    combined = pd.concat([dfA[descA], dfB[descB]], ignore_index=True)
                    if combined.empty:
                        st.warning("Descriptions are empty after cleaning.")
                    else:
                        vec.fit(combined.tolist())
                        XA = vec.transform(dfA[descA].tolist())
                        XB = vec.transform(dfB[descB].tolist())

                        if XA.shape[0] == 0 or XB.shape[0] == 0:
                            st.warning("No rows to compare after cleaning.")
                        else:
                            sim = cosine_similarity(XA, XB)
                            pairs = _cross_long(sim, ids_a=dfA[idA].tolist(), ids_b=dfB[idB].tolist())
                            mapA = dict(zip(dfA[idA], dfA[descA]))
                            mapB = dict(zip(dfB[idB], dfB[descB]))
                            pairs["Desc_A"] = pairs["ID_A"].map(mapA)
                            pairs["Desc_B"] = pairs["ID_B"].map(mapB)

                            pairs = pairs[["Story A ID", "Story A Desc", "Story BI ", "Story B Desc", "Similarity Score"]]
                            if threshold > 0:
                                pairs = pairs[pairs["similarity"] >= threshold]
                            if topk_per_A > 0:
                                pairs = (
                                    pairs.sort_values("similarity", ascending=not sort_desc)
                                    .groupby("ID_A", as_index=False, sort=False)
                                    .head(topk_per_A)
                                )
                            elif sort_desc:
                                pairs = pairs.sort_values("similarity", ascending=False)

                            st.success(f"Done. {len(pairs):,} A×B pairs.")
                            st.dataframe(pairs.head(show_preview_rows), use_container_width=True)

                            meta = {
                                "Build UTC": BUILD_TS,
                                "Mode": "Two files (A vs B)",
                                "File A IDs": idA,
                                "File A Desc": descA,
                                "File B IDs": idB,
                                "File B Desc": descB,
                                "Threshold": threshold,
                                "TopK per A": topk_per_A,
                                "Sort desc": sort_desc,
                                "n-gram": f"{ngram_min}-{ngram_max}",
                                "min_df": min_df,
                                "max_df": max_df,
                            }

                            c1, c2 = st.columns(2)
                            with c1:
                                st.download_button(
                                    "⬇️ Download CSV results",
                                    data=pairs.to_csv(index=False).encode("utf-8"),
                                    file_name="similarity_pairs_A_vs_B.csv",
                                    mime="text/csv",
                                )
                            with c2:
                                xlsx_bytes = downloadable_excel(pairs, sheet_name="pairs_A_vs_B", meta=meta)
                                if xlsx_bytes:
                                    st.download_button(
                                        "⬇️ Download Excel results",
                                        data=xlsx_bytes,
                                        file_name="similarity_pairs_A_vs_B.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    )
                                else:
                                    st.caption("ℹ️ Excel export unavailable (install `xlsxwriter` or `openpyxl`). CSV works.")

# -------------------------
# Friendly footer / FAQ
# -------------------------
with st.expander("Need help? (FAQ)"):
    st.markdown(
        "- **What is TF-IDF?** Turns text into weighted token vectors; common words matter less.\n"
        "- **Similarity:** Cosine ∈ [0, 1]; higher = closer.\n"
        "- **Threshold:** Hide pairs below a cutoff.\n"
        "- **Top-K per ID_A:** Keep only K best matches for each source row.\n"
        "- **n-grams:** Include short phrases (e.g., 2-grams like “user login”)."
    )
