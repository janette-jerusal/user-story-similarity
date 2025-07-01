import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import base64
from io import BytesIO
st.set_page_config(page_title="User Story Similarity App", layout="wide")
st.title("📘 User Story Similarity Checker")
st.markdown("Upload **one or two Excel files** with user stories. The app will compute similarities and return top matches.")
uploaded_files = st.file_uploader("Upload 1 or 2 Excel files", type=["xlsx"], accept_multiple_files=True)
def detect_columns(df):
   id_col = None
   desc_col = None
   for col in df.columns:
       col_lower = col.lower()
       if 'id' in col_lower:
           id_col = col
       if 'desc' in col_lower:
           desc_col = col
   return id_col, desc_col
def process_and_match(df1, df2, id1, desc1, id2, desc2):
   tfidf = TfidfVectorizer().fit(pd.concat([df1[desc1], df2[desc2]]))
   tfidf1 = tfidf.transform(df1[desc1])
   tfidf2 = tfidf.transform(df2[desc2])
   similarity = cosine_similarity(tfidf1, tfidf2)
   results = []
   for i in range(similarity.shape[0]):
       top_indices = similarity[i].argsort()[-3:][::-1]
       for rank, j in enumerate(top_indices, 1):
           results.append({
               "Story A ID": df1[id1].iloc[i],
               "Story A Desc": df1[desc1].iloc[i],
               "Story B ID": df2[id2].iloc[j],
               "Story B Desc": df2[desc2].iloc[j],
               "Similarity Score (%)": round(similarity[i][j] * 100, 2),
               "Match Rank": rank
           })
   return pd.DataFrame(results)
def create_download_link(df):
   output = BytesIO()
   with pd.ExcelWriter(output, engine="openpyxl") as writer:
       df.to_excel(writer, index=False, sheet_name="Top Matches")
   output.seek(0)
   b64 = base64.b64encode(output.read()).decode()
   return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="user_story_matches.xlsx">📥 Download Results as Excel</a>'
if uploaded_files:
   if len(uploaded_files) == 1:
       df = pd.read_excel(uploaded_files[0])
       source_name = uploaded_files[0].name
       id_col, desc_col = detect_columns(df)
       if not id_col or not desc_col:
           st.error("Could not detect 'ID' and 'Description' columns.")
       else:
           df['source'] = source_name
           results_df = process_and_match(df, df, id_col, desc_col, id_col, desc_col)
   elif len(uploaded_files) == 2:
       df1 = pd.read_excel(uploaded_files[0])
       df2 = pd.read_excel(uploaded_files[1])
       id1, desc1 = detect_columns(df1)
       id2, desc2 = detect_columns(df2)
       if not id1 or not desc1 or not id2 or not desc2:
           st.error("Could not detect 'ID' and 'Description' columns in one or both files.")
       else:
           df1['source'] = uploaded_files[0].name
           df2['source'] = uploaded_files[1].name
           results_df = process_and_match(df1, df2, id1, desc1, id2, desc2)
   else:
       st.error("Please upload only 1 or 2 files.")
       st.stop()
   if 'results_df' in locals() and not results_df.empty:
       st.success("✅ Match results ready!")
       st.dataframe(results_df)
       st.markdown(f"**Total Unique User Stories A:** {results_df['Story A ID'].nunique()}")
       st.markdown(f"**Total Unique Matches Found:** {len(results_df)}")
       # Visualization
       match_counts = results_df['Story A ID'].value_counts().value_counts().sort_index()
       fig, ax = plt.subplots()
       match_counts.plot(kind='bar', ax=ax)
       ax.set_xlabel("Number of Matches per Story A")
       ax.set_ylabel("Count")
       ax.set_title("Distribution of Match Frequency")
       st.pyplot(fig)
       # Download link
       st.markdown(create_download_link(results_df), unsafe_allow_html=True)
