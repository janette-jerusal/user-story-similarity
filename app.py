import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
st.set_page_config(page_title="User Story Similarity Tool", layout="wide")
st.title("📌 User Story Similarity Checker")
st.markdown("Upload **1 or 2 Excel files** to compare user stories and identify similar ones.")
# Upload files
uploaded_files = st.file_uploader(
   "Upload 1 or 2 Excel files",
   type=["xlsx"],
   accept_multiple_files=True,
   key="file_uploader_unique"
)
if uploaded_files:
   dfs = []
   for uploaded_file in uploaded_files:
       df = pd.read_excel(uploaded_file)
       file_label = uploaded_file.name
       df['SourceFile'] = file_label
       dfs.append(df)
   # Combine and validate columns
   combined_df = pd.concat(dfs, ignore_index=True)
   st.subheader("✅ File(s) Uploaded Successfully")
   # Let user define column names
   col1, col2 = st.columns(2)
   with col1:
       id_col = st.selectbox("Select the ID column", combined_df.columns)
   with col2:
       desc_col = st.selectbox("Select the Description column", combined_df.columns)
   # Proceed if 1 or more stories exist
   if len(combined_df) > 1:
       descriptions = combined_df[desc_col].astype(str).tolist()
       vectorizer = TfidfVectorizer()
       tfidf_matrix = vectorizer.fit_transform(descriptions)
       cosine_sim = cosine_similarity(tfidf_matrix)
       matches = []
       for i in range(len(combined_df)):
           for j in range(i + 1, len(combined_df)):
               sim_score = cosine_sim[i][j]
               matches.append({
                   "Story A ID": combined_df.iloc[i][id_col],
                   "Story A Desc": combined_df.iloc[i][desc_col],
                   "Story A Source": combined_df.iloc[i]["SourceFile"],
                   "Story B ID": combined_df.iloc[j][id_col],
                   "Story B Desc": combined_df.iloc[j][desc_col],
                   "Story B Source": combined_df.iloc[j]["SourceFile"],
                   "Similarity Score": round(sim_score, 4)
               })
       result_df = pd.DataFrame(matches)
       result_df = result_df.sort_values(by="Similarity Score", ascending=False).reset_index(drop=True)
       # KPIs
       st.subheader("📊 Summary")
       col1, col2 = st.columns(2)
       col1.metric("Total User Stories", len(combined_df))
       col2.metric("Matched Pairs (Score > 0.5)", len(result_df[result_df["Similarity Score"] > 0.5]))
       # Show results table
       st.subheader("🔍 Top Matches")
       st.dataframe(result_df.head(10))
       # Excel export
       def convert_df_to_excel(df):
           output = BytesIO()
           with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
               df.to_excel(writer, index=False, sheet_name='Matches')
           output.seek(0)
           return output
       excel_data = convert_df_to_excel(result_df)
       st.download_button(
           label="📥 Download Results as Excel",
           data=excel_data,
           file_name="user_story_matches.xlsx",
           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
       )
   else:
       st.warning("Please upload at least 2 user stories to compare.")
