import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="User Story Similarity Tool", layout="wide")
st.title("User Story Similarity Comparison")

st.markdown("""
This tool allows you to upload two Excel files containing user stories (e.g., Initiate and Manage contracts).
It will compare the descriptions and return matches above a selected similarity threshold.
""")

# Upload files
file1 = st.file_uploader("Upload First Excel File (Initiate User Stories)", type=["xlsx"])
file2 = st.file_uploader("Upload Second Excel File (Manage User Stories)", type=["xlsx"])

# Set similarity threshold
threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

# Optional column selector
st.markdown("### Column Selection")
desc_col1 = st.text_input("Description Column Name in First File", value="Description")
desc_col2 = st.text_input("Description Column Name in Second File", value="Description")
id_col1 = st.text_input("ID Column Name in First File", value="ID")
id_col2 = st.text_input("ID Column Name in Second File", value="ID")

if file1 and file2:
    try:
        df1 = pd.read_excel(file1)
        df2 = pd.read_excel(file2)

        # Drop NA
        df1 = df1.dropna(subset=[desc_col1])
        df2 = df2.dropna(subset=[desc_col2])

        # Vectorize descriptions
        combined = pd.concat([df1[desc_col1], df2[desc_col2]])
        vectorizer = TfidfVectorizer().fit(combined)
        tfidf_1 = vectorizer.transform(df1[desc_col1])
        tfidf_2 = vectorizer.transform(df2[desc_col2])

        # Compute cosine similarity
        sim_matrix = cosine_similarity(tfidf_1, tfidf_2)

        # Build matches
        matches = []
        for i in range(sim_matrix.shape[0]):
            for j in range(sim_matrix.shape[1]):
                score = sim_matrix[i, j]
                if score >= threshold:
                    matches.append({
                        "Initiate ID": df1.iloc[i][id_col1],
                        "Manage ID": df2.iloc[j][id_col2],
                        "Initiate Description": df1.iloc[i][desc_col1],
                        "Manage Description": df2.iloc[j][desc_col2],
                        "Similarity Score": round(score, 2)
                    })

        result_df = pd.DataFrame(matches)

        if not result_df.empty:
            st.success(f"Found {len(result_df)} matches above the threshold.")
            st.dataframe(result_df)
            st.download_button("Download Results as Excel", data=result_df.to_excel(index=False), file_name="similar_user_stories.xlsx")
        else:
            st.warning("No matches found above the selected threshold.")

    except Exception as e:
        st.error(f"Error: {e}")
