# app.py — Keyword Search Tool (Duplicate-column safe)

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Keyword Search", layout="wide")

st.title("Keyword Search Tool")

# =====================================================
# Helpers
# =====================================================

def ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Force unique column names (required by PyArrow)."""
    df = df.copy()
    seen = {}
    new_cols = []

    for col in df.columns:
        col = str(col).strip()
        if col not in seen:
            seen[col] = 1
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}__{seen[col]}")

    df.columns = new_cols
    return df


def safe_dataframe(df: pd.DataFrame):
    """
    Safely display a dataframe in Streamlit.
    Removes duplicate columns before rendering.
    """
    df = ensure_unique_columns(df)
    df = df.loc[:, ~df.columns.duplicated()]
    st.dataframe(df, use_container_width=True)


# =====================================================
# File Upload
# =====================================================

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:

    # Read file
    try:
        if uploaded_file.name.endswith(("xlsx", "xls")):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    df = ensure_unique_columns(df)

    st.subheader("Preview (First 10 Rows)")
    safe_dataframe(df.head(10))

    st.divider()

    # =====================================================
    # Column Selection
    # =====================================================

    st.subheader("Select Columns")

    columns = list(df.columns)

    keyword_col = st.selectbox("Keyword Column", columns)
    retain_cols = st.multiselect(
        "Columns to Display",
        columns,
        default=columns
    )

    # Remove duplicates from retain_cols (just in case)
    retain_cols = list(dict.fromkeys(retain_cols))

    # =====================================================
    # Filtering
    # =====================================================

    st.subheader("Keyword Filter")

    search_term = st.text_input("Enter keyword to search")

    if search_term:
        filtered_df = df[df[keyword_col].astype(str).str.contains(
            search_term,
            case=False,
            na=False
        )]
    else:
        filtered_df = df.copy()

    # Ensure retain_cols exist
    retain_cols = [c for c in retain_cols if c in filtered_df.columns]

    combined_df = filtered_df[retain_cols].copy()

    # 🔥 CRITICAL FIX: ensure no duplicate columns before displaying
    combined_df = ensure_unique_columns(combined_df)
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

    st.subheader("Filtered Results (Top 10)")
    safe_dataframe(combined_df.head(10))

    # =====================================================
    # Download Option
    # =====================================================

    st.download_button(
        "Download Filtered CSV",
        combined_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_keywords.csv",
        mime="text/csv"
    )

else:
    st.info("Upload a file to begin.")
