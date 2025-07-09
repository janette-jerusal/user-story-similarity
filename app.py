import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="User Story Similarity", layout="wide")

# ---------- File Upload ----------
st.title("🔍 User Story Similarity Checker")

file1 = st.file_uploader("Upload File 1", type=["csv", "xlsx"])
file2 = st.file_uploader("Upload File 2", type=["csv", "xlsx"])

similarity_threshold = st.slider("Set Similarity Threshold", 0.0, 1.0, 0.65, step=0.01)

def read_file(file):
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        df.columns = df.columns.str.strip().str.lower()
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def compute_similarity(df1, df2, threshold):
    combined = df1["desc"].tolist() + df2["desc"].tolist()
    tfidf = TfidfVectorizer().fit_transform(combined)
    
    sim_matrix = cosine_similarity(tfidf[:len(df1)], tfidf[len(df1):])
    
    max_sim_idx = sim_matrix.argmax(axis=1)
    max_sim_values = sim_matrix.max(axis=1)
    
    result = pd.DataFrame({
        "File1_ID": df1["id"],
        "File1_Desc": df1["desc"],
        "BestMatch_File2_ID": df2.loc[max_sim_idx, "id"].values,
        "BestMatch_File2_Desc": df2.loc[max_sim_idx, "desc"].values,
        "Similarity": max_sim_values
    })
    
    return result[result["Similarity"] >= threshold].reset_index(drop=True), sim_matrix

# ---------- Process Button ----------
if file1 and file2:
    df1 = read_file(file1)
    df2 = read_file(file2)

    if df1 is not None and df2 is not None:
        required_cols = {"id", "desc"}
        if not required_cols.issubset(df1.columns):
            st.error("❌ File 1 must contain 'id' and 'desc' columns.")
        elif not required_cols.issubset(df2.columns):
            st.error("❌ File 2 must contain 'id' and 'desc' columns.")
        else:
            st.success("✅ Files successfully loaded!")
            if st.button("🔍 Compare"):
                with st.spinner("Computing similarities..."):
                    results_df, sim_matrix = compute_similarity(df1, df2, similarity_threshold)

                st.subheader("🔗 Matching Results")
                st.dataframe(results_df, use_container_width=True)

                st.subheader("📊 Similarity Distribution")
                fig, ax = plt.subplots()
                sns.histplot(results_df["Similarity"], bins=20, kde=True, ax=ax)
                ax.axvline(similarity_threshold, color='red', linestyle='--', label='Threshold')
                ax.set_title("Similarity Score Distribution")
                ax.set_xlabel("Similarity Score")
                ax.set_ylabel("Frequency")
                ax.legend()
                st.pyplot(fig)

                st.subheader("📈 KPIs")
                total = len(df1)
                matched = len(results_df)
                unmatched = total - matched
                avg_similarity = results_df["Similarity"].mean() if matched > 0 else 0.0

                col1, col2, col3 = st.columns(3)
                col1.metric("✅ Matched", f"{matched} / {total}")
                col2.metric("❌ Unmatched", unmatched)
                col3.metric("📈 Avg Similarity", f"{avg_similarity:.2f}")

