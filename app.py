import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Streamlit App UI
st.title("User Story Similarity Comparator")
st.write("Upload two Excel files containing user stories (must have 'id' and 'desc' columns).")

uploaded_file_1 = st.file_uploader("Upload File 1", type=["xlsx"])
uploaded_file_2 = st.file_uploader("Upload File 2", type=["xlsx"])

if uploaded_file_1 and uploaded_file_2:
    df1 = pd.read_excel(uploaded_file_1, header=0)
    df2 = pd.read_excel(uploaded_file_2, header=0)

    # Normalize column names
    df1.columns = df1.columns.str.strip().str.lower()
    df2.columns = df2.columns.str.strip().str.lower()

    if "id" not in df1.columns or "desc" not in df1.columns:
        st.error("File 1 must contain 'id' and 'desc' columns.")
    elif "id" not in df2.columns or "desc" not in df2.columns:
        st.error("File 2 must contain 'id' and 'desc' columns.")
    else:
        # Vectorization
        all_descriptions = df1["desc"].tolist() + df2["desc"].tolist()
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_descriptions)

        tfidf_1 = tfidf_matrix[:len(df1)]
        tfidf_2 = tfidf_matrix[len(df1):]

        # Cosine similarity
        similarity_matrix = cosine_similarity(tfidf_1, tfidf_2)

        # Build similarity DataFrame
        result_data = []
        for i, id1 in enumerate(df1["id"]):
            for j, id2 in enumerate(df2["id"]):
                result_data.append({
                    "ID File 1": id1,
                    "ID File 2": id2,
                    "Similarity": similarity_matrix[i][j]
                })

        result_df = pd.DataFrame(result_data)

        # Show top 10 most similar
        st.subheader("Top 10 Most Similar User Stories")
        st.dataframe(result_df.sort_values(by="Similarity", ascending=False).head(10))

        # KPI: Average Similarity
        avg_similarity = result_df["Similarity"].mean()
        st.metric("Average Similarity Score", f"{avg_similarity:.2f}")

        # Heatmap
        st.subheader("Similarity Heatmap")
        plt.figure(figsize=(10, 6))
        sns.heatmap(similarity_matrix, cmap="YlGnBu", xticklabels=df2["id"], yticklabels=df1["id"])
        st.pyplot(plt)
else:
    st.warning("Please upload both files to proceed.")

