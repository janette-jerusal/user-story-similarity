import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
st.set_page_config(page_title="User Story Similarity Matcher", layout="wide")
st.title("📄 User Story Similarity Matcher")
st.markdown("Upload **two Excel files** with user stories. Select the **ID** and **description** columns for each file. The app finds the **top 3 most similar stories (≥ 70%)** from the second file for each story in the first.")
# Upload files
file1 = st.file_uploader("Upload First File", type=["xlsx"])
file2 = st.file_uploader("Upload Second File", type=["xlsx"])
if file1 and file2:
   df1 = pd.read_excel(file1)
   df2 = pd.read_excel(file2)
   st.subheader("📌 Select Columns for First File")
   id_col_1 = st.selectbox("Select ID column (File 1)", df1.columns, key="id1")
   desc_col_1 = st.selectbox("Select Description column (File 1)", df1.columns, key="desc1")
   st.subheader("📌 Select Columns for Second File")
   id_col_2 = st.selectbox("Select ID column (File 2)", df2.columns, key="id2")
   desc_col_2 = st.selectbox("Select Description column (File 2)", df2.columns, key="desc2")
   if st.button("🔍 Find Top 3 Matches"):
       try:
           # TF-IDF vectorization
           vectorizer = TfidfVectorizer(stop_words="english")
           tfidf_1 = vectorizer.fit_transform(df1[desc_col_1].astype(str))
           tfidf_2 = vectorizer.transform(df2[desc_col_2].astype(str))
           sim_matrix = cosine_similarity(tfidf_1, tfidf_2)
           # Match results
           results = []
           match_counts = {}
           all_scores = []
           for i, row in df1.iterrows():
               top_indices = sim_matrix[i].argsort()[::-1][:3]
               for idx in top_indices:
                   score = sim_matrix[i][idx]
                   all_scores.append(score)
                   if score >= 0.7:
                       match_counts[row[id_col_1]] = match_counts.get(row[id_col_1], 0) + 1
                       results.append({
                           "File 1 ID": row[id_col_1],
                           "File 1 Desc": row[desc_col_1],
                           "File 2 ID": df2.iloc[idx][id_col_2],
                           "File 2 Desc": df2.iloc[idx][desc_col_2],
                           "Similarity Score (%)": round(score * 100, 2)
                       })
           if results:
               results_df = pd.DataFrame(results)
               # Summary dashboard
               total_file1 = df1.shape[0]
               matched_stories = len(set(match_counts.keys()))
               total_matches = len(results_df)
               st.subheader("📌 Summary")
               col1, col2, col3 = st.columns(3)
               col1.metric("Stories in File 1", total_file1)
               col2.metric("Stories Matched", matched_stories)
               col3.metric("Total Matches", total_matches)
               # Results table
               st.subheader("📋 Matching Results")
               st.dataframe(results_df)
               # Histogram of scores
               st.subheader("📊 Similarity Score Distribution")
               fig, ax = plt.subplots()
               ax.hist([s * 100 for s in all_scores], bins=20, color='skyblue', edgecolor='black')
               ax.set_xlabel("Similarity Score (%)")
               ax.set_ylabel("Frequency")
               ax.set_title("Histogram of All Similarity Scores")
               st.pyplot(fig)
               # Bar chart of match counts
               st.subheader("📈 Number of Matches per Story (≥ 70%)")
               match_df = pd.DataFrame({
                   "User Story ID": list(match_counts.keys()),
                   "Number of Matches": list(match_counts.values())
               })
               st.bar_chart(match_df.set_index("User Story ID"))
               # Excel download
               excel_output = io.BytesIO()
               with pd.ExcelWriter(excel_output, engine='openpyxl') as writer:
                   results_df.to_excel(writer, index=False, sheet_name="Top 3 Matches")
               st.download_button(
                   label="⬇️ Download Results as Excel",
                   data=excel_output.getvalue(),
                   file_name="Top3_Matches_User_Stories.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
               )
           else:
               st.warning("⚠️ No matches with similarity ≥ 70%. Try different columns or files.")
       except Exception as e:
           st.error(f"🚨 Error: {e}")
