import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

st.set_page_config(page_title="User‑Story Similarity Comparator", layout="wide")
st.title("📊 User‑Story Similarity Comparator")

# 1️⃣ Upload Files
file1 = st.file_uploader("Upload file 1 (CSV / Excel)", key="f1")
file2 = st.file_uploader("Upload file 2 (CSV / Excel, optional)", key="f2")
threshold = st.slider("Similarity threshold (%)", 0, 100, 60, 1)

# 2️⃣ File Loader
def load_file(f):
    if f is None:
        return None
    try:
        if f.name.endswith(".csv"):
            return pd.read_csv(f)
        elif f.name.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(f.read()))
        else:
            raise ValueError("Unsupported file type.")
    except Exception as e:
        raise ValueError(f"Failed to load file '{f.name}': {e}")

# 3️⃣ Preload and let user choose column names
df1 = load_file(file1) if file1 else None
df2 = load_file(file2) if file2 else None

if df1 is not None:
    st.subheader("Step 1: Select Columns for File 1")
    col_id_1 = st.selectbox("Select ID column for file 1", df1.columns, key="id1")
    col_desc_1 = st.selectbox("Select description column for file 1", df1.columns, key="desc1")

if df2 is not None:
    st.subheader("Step 2: Select Columns for File 2")
    col_id_2 = st.selectbox("Select ID column for file 2", df2.columns, key="id2")
    col_desc_2 = st.selectbox("Select description column for file 2", df2.columns, key="desc2")
elif df1 is not None:
    col_id_2 = col_id_1
    col_desc_2 = col_desc_1
    df2 = df1.copy()

# 4️⃣ Similarity Computation
def compute_similarity(df1, df2, col_id_1, col_desc_1, col_id_2, col_desc_2, thr):
    df1 = df1[[col_id_1, col_desc_1]].rename(columns={col_id_1: "id", col_desc_1: "desc"})
    df2 = df2[[col_id_2, col_desc_2]].rename(columns={col_id_2: "id", col_desc_2: "desc"})
    
    df1["desc"] = df1["desc"].fillna("").astype(str)
    df2["desc"] = df2["desc"].fillna("").astype(str)

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

# 5️⃣ Final Comparison Trigger
if st.button("🔍 Compare") and df1 is not None and col_id_1 and col_desc_1:
    try:
        result = compute_similarity(df1, df2, col_id_1, col_desc_1, col_id_2, col_desc_2, threshold)

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
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")

st.caption("Made with ❤️ & Streamlit")
