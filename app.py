# Housekeeping 
import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

# Page
st.set_page_config(page_title="User Story Consolidator & Matcher", layout="wide")
st.title("User Story Analysis Tool")


# Upload multiple files
uploaded_files = st.file_uploader("Upload One or More Excel Files", type=["xlsx"], accept_multiple_files=True)

# User inputs for column names
desc_col = st.text_input("Description Column Name", value="Desc")
id_col = st.text_input("ID Column Name", value="ID")

# Similarity threshold
threshold = st.slider("Similarity Threshold (%)", 0, 100, 70)
if uploaded_files:

    try:
        all_data = []
        # Loading files and tagging with its file name
        for uploaded_file in uploaded_files:
            df = pd.read_excel(uploaded_file)
            df["Source File"] = os.path.basename(uploaded_file.name)
            all_data.append(df)

        # Combine into a single DataFrame
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df[[id_col, desc_col, "Source File"]].dropna()

        st.subheader("Consolidated Data")
        st.dataframe(combined_df)
       
        # Text stuff
        descriptions = combined_df[desc_col].astype(str).fillna("")
        tfidf = TfidfVectorizer().fit(descriptions.tolist())
        tfidf_matrix = tfidf.transform(descriptions)

        # Cosine similarity with full dataset
        similarity_matrix = cosine_similarity(tfidf_matrix)
        matches = []
        for i in range(len(descriptions)):
            for j in range(i + 1, len(descriptions)):
                score = similarity_matrix[i][j]
                if score * 100 >= threshold:
                    matches.append({
                        "Story A ID": combined_df[id_col].iloc[i],
                        "Story A Desc": descriptions.iloc[i],
                        "Story A Source": combined_df["Source File"].iloc[i],
                        "Story B ID": combined_df[id_col].iloc[j],
                        "Story B Desc": descriptions.iloc[j],
                        "Story B Source": combined_df["Source File"].iloc[j],
                        "Similarity %": round(score * 100, 2)
                    })

        results_df = pd.DataFrame(matches)
        if not results_df.empty:
            st.subheader("Matching User Stories")
            st.dataframe(results_df)
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                results_df.to_excel(writer, index=False, sheet_name="Matches")
            buffer.seek(0)
            st.download_button(
                label="Download Results as Excel",
                data=buffer,
                file_name="User_Story_Matches.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("No matches found. Try lowering the threshold.")
    except Exception as e:
        st.error(f"Error: {e}")
 
