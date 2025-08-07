import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="User‑Story Similarity Comparator", layout="wide")
st.title("📊 User‑Story Similarity Comparator")

# ----------------------------------------------------------
# 1️⃣ Upload widgets
# ----------------------------------------------------------
file1 = st.file_uploader("Upload file 1 (CSV / Excel)", key="f1")
file2 = st.file_uploader("Upload file 2 (CSV / Excel, optional)", key="f2")
threshold = st.slider("Similarity threshold (%)", 0, 100, 60, 1)

def load_file(f):
    if f is None:
        return None
    try:
        if f.name.endswith(".csv"):
            return pd.read_csv(f)
        return pd.read_excel(f)
    except Exception as e:
        st.error(f"Error loading {f.name}: {e}")
        return None

# ----------------------------------------------------------
# 2️⃣ Comparison logic
# ----------------------------------------------------------
def compute_similarity(df1: pd.DataFrame, df2: pd.DataFrame, thr: float):
    df1 = df1.rename(columns=str.lower)
    df2 = df2.rename(columns=str.lower)

    for df in (df1, df2):
        if df is None:
            raise ValueError("One of the files could not be read.")
        if {"id", "desc"} - set(df.columns):
            raise ValueError("Each file must include both 'id' and 'desc' columns.")
        df["desc"] = df["desc"].fillna("").astype(str)

    combined = pd.concat([df1["desc"], df2["desc"]]).values
    tfidf = TfidfVectorizer().fit_transform(combined)

    tfidf_1 = tfidf[: len(df1)]
    tfidf_2 = tfidf[len(df1):]

    sim_matrix = cosine_similarity(tfidf_1, tfidf_2)
    matches = []
    for i, (id1, d1) in enumerate(zip(df1["id"], df1["desc"])):
        for j, (id2, d2) in enumerate(zip(df2["id"], df2["desc"])):
            sim = sim_matrix[i, j]
            if sim * 100 >= thr:
                matches.append({
                    "id_1": id1,
                    "id_2": id2,
                    "similarity_%": sim * 100,
                })
    return pd.DataFrame(matches)

# ----------------------------------------------------------
# 3️⃣ Button & results
# ----------------------------------------------------------
if st.button("🔍 Compare"):
    if not file1:
        st.warning("Please upload at least one file.")
    else:
        try:
            df1 = load_file(file1)
            if file2:
                df2 = load_file(file2)
            else:
                # Reload file1 from raw bytes to prevent Streamlit read issue
                file1_bytes = file1.read()
                file1.seek(0)  # Reset pointer so Streamlit can re-read it
                if file1.name.endswith(".csv"):
                    df2 = pd.read_csv(pd.io.common.BytesIO(file1_bytes))
                else:
                    df2 = pd.read_excel(pd.io.common.BytesIO(file1_bytes))

            result = compute_similarity(df1, df2, threshold)

            st.success(f"Comparison finished. {len(result)} matching pairs found ✅")

            col1, col2, col3 = st.columns(3)
            total_pairs = len(df1) * len(df2)
            match_ratio = (len(result) / total_pairs * 100) if total_pairs else 0
            avg_sim = result["similarity_%"].mean() if len(result) else 0

            col1.metric("🎯 Total Stories Compared", f"{total_pairs:,}")
            col2.metric("✅ # Matches", f"{len(result):,}", f"{match_ratio:.1f}% of pairs")
            col3.metric("📈 Avg Similarity", f"{avg_sim:.1f}%")

            if len(result):
                st.dataframe(
                    result.sort_values("similarity_%", ascending=False)
                          .style.bar("similarity_%", vmax=100, color="#5fba7d")
                          .format({"similarity_%": "{:.1f} %"}),
                    height=400,
                )
            else:
                st.info("No matches above the selected threshold.")

        except ValueError as ve:
            st.error(f"❌ {ve}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

st.caption("Made with ❤️ & Streamlit")

