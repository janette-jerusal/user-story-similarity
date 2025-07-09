import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(page_title="User Story Similarity Checker", layout="wide")

st.title("📊 User Story Similarity Checker")

# Upload files
uploaded_file_1 = st.file_uploader("Upload File 1 (Excel with 'id' and 'desc')", type=["xlsx"])
uploaded_file_2 = st.file_uploader("Upload File 2 (Excel with 'id' and 'desc')", type=["xlsx"])

# Set threshold
similarity_threshold = st.slider("Set Similarity Threshold", min_value=0.0, max_value=1.0, value=0.65, step=0.01)

def compute_similarity(df1, df2, threshold):
    df1 = df1.dropna(subset=["desc"])
    df2 = df2.dropna(subset=["desc"])

    combined_desc = pd.concat([df1["desc"], df2["desc"]], ignore_index=True)
    tfidf = TfidfVectorizer().fit_transform(combined_desc)

    tfidf_1 = tfidf[:len(df1)]
    tfidf_2 = tfidf[len(df1):]

    sim_matrix = cosine_similarity(tfidf_1, tfidf_2)

    matches = []
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            sim_score = sim_matrix[i, j]
            if sim_score >= threshold:
                matches.append({
                    "File1_ID": df1.iloc[i]["id"],
                    "File1_Desc": df1.iloc[i]["desc"],
                    "File2_ID": df2.iloc[j]["id"],
                    "File2_Desc": df2.iloc[j]["desc"],
                    "Similarity": round(sim_score, 4)
                })

    return pd.DataFrame(matches)

# Run comparison
if uploaded_file_1 and uploaded_file_2:
    try:
        df1 = pd.read_excel(uploaded_file_1)
        df2 = pd.read_excel(uploaded_file_2)

        if "id" not in df1.columns or "desc" not in df1.columns:
            st.error("File 1 must contain 'id' and 'desc' columns.")
        elif "id" not in df2.columns or "desc" not in df2.columns:
            st.error("File 2 must contain 'id' and 'desc' columns.")
        else:
            result_df = compute_similarity(df1, df2, similarity_threshold)

            st.success(f"Found {len(result_df)} similar pairs")

            # KPIs
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("File 1 Records", len(df1))
            col2.metric("File 2 Records", len(df2))
            col3.metric("Matches", len(result_df))
            col4.metric("Avg Similarity", round(result_df["Similarity"].mean(), 4) if not result_df.empty else 0)
            col5.metric("Max Similarity", round(result_df["Similarity"].max(), 4) if not result_df.empty else 0)
            col6.metric("Min Similarity", round(result_df["Similarity"].min(), 4) if not result_df.empty else 0)

            st.dataframe(result_df)

            # Optional: Download result
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results as CSV", csv, "similarities.csv", "text/csv")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
