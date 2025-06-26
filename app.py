import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

results_df = pd.DataFrame()

# Page Config 
st.set_page_config(page_title="User Story Matcher", layout="wide")
st.markdown("""
# User Story Similarity Matcher
Upload multiple Excel files of user stories, match similar entries, and download results.
""")

# File Upload
with st.expander("Upload Files ", expanded=True):
    uploaded_files = st.file_uploader(
        "Upload one or more `.xlsx` files (e.g., Initiate or Manage user stories)",
        type=["xlsx"],
        accept_multiple_files=True,
        help="You can upload several Excel files. Each will be tagged by its file name."
    )

# Column Names
st.markdown("### Column Settings")
desc_col = st.text_input("Description Column Name", value="Desc", help="Column containing the user story text.")
id_col = st.text_input("ID Column Name", value="ID", help="Unique identifier for each user story).")

# Similarity Threshold
threshold = st.slider("Similarity Threshold (%)", 0, 100, 70, help="Only story pairs with similarity above this threshold will appear in the results.")

# Processing
if uploaded_files:
    try:
        with st.spinner("... reviewing matches "):
            all_data = []
            for uploaded_file in uploaded_files:
                df = pd.read_excel(uploaded_file)
                df["Source File"] = os.path.basename(uploaded_file.name)
                all_data.append(df)
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df[[id_col, desc_col, "Source File"]].dropna()
            combined_df[desc_col] = combined_df[desc_col].astype(str)
            st.markdown("Combined User Stories")
            st.dataframe(combined_df)
            descriptions = combined_df[desc_col]
            tfidf = TfidfVectorizer().fit(descriptions.tolist())
            tfidf_matrix = tfidf.transform(descriptions)
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
                st.markdown("Matching Results")
                st.success(f"{len(results_df)} matching pairs found above {threshold}% similarity.")
                st.dataframe(results_df)
                
                # Excel Output
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
                st.warning("No matches found above this threshold. Adjust threshold.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Visuals
if not results_df.empty:
   st.markdown("Highlights of your Selected Comparison Report")
   total_stories = len(combined_df)
   total_matches = len(results_df)
   uploaded_file_names = combined_df["Source File"].unique()
   # Metrics
   col1, col2, col3 = st.columns(3)
   col1.metric("Total User Stories", total_stories)
   col2.metric("Matched Pairs", total_matches)
   col3.metric("Files Uploaded", len(uploaded_file_names))
   # Bar Chart
   match_counts = results_df["Story A Source"].value_counts().add(results_df["Story B Source"].value_counts(), fill_value=0)
   st.bar_chart(match_counts)
   # Top Match
   top_match = results_df.sort_values(by="Similarity %", ascending=False).iloc[0]
   with st.expander("Top Match", expanded=True):
       st.markdown(f"""
       **Similarity:** {top_match['Similarity %']}%  
       - **Story A:** `{top_match['Story A ID']}` from `{top_match['Story A Source']}`  
       - **Story B:** `{top_match['Story B ID']}` from `{top_match['Story B Source']}`  
       - **Story A Desc:** {top_match['Story A Desc']}  
       - **Story B Desc:** {top_match['Story B Desc']}  
       """)
