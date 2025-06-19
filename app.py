import streamlit as st
import pandas as pd
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
st.title("📚 Similar User Stories Matcher")
st.markdown("Upload two Excel files of user stories to compare and find similar descriptions.")
# Upload files
file1 = st.file_uploader("Upload First Excel File", type=["xlsx"])
file2 = st.file_uploader("Upload Second Excel File", type=["xlsx"])
# Text inputs for column names
desc_col1 = st.text_input("Description Column Name in First File", "Desc")
desc_col2 = st.text_input("Description Column Name in Second File", "Desc")
id_col1 = st.text_input("ID Column Name in First File", "ID")
id_col2 = st.text_input("ID Column Name in Second File", "ID")
# Similarity threshold
threshold = st.slider("Similarity Threshold (0-1)", 0.0, 1.0, 0.7, 0.05)
if file1 and file2:
   try:
       df1 = pd.read_excel(file1, engine="openpyxl")
       df2 = pd.read_excel(file2, engine="openpyxl")
       # Extract relevant columns
       stories1 = df1[[id_col1, desc_col1]].dropna()
       stories2 = df2[[id_col2, desc_col2]].dropna()
       # Vectorize descriptions
       tfidf = TfidfVectorizer().fit(pd.concat([stories1[desc_col1], stories2[desc_col2]]))
       vec1 = tfidf.transform(stories1[desc_col1])
       vec2 = tfidf.transform(stories2[desc_col2])
       # Calculate cosine similarity
       similarity = cosine_similarity(vec1, vec2)
       matches = []
       for i in range(similarity.shape[0]):
           for j in range(similarity.shape[1]):
               score = similarity[i, j]
               if score >= threshold:
                   matches.append({
                       f"{id_col1}": stories1.iloc[i][id_col1],
                       f"{desc_col1}": stories1.iloc[i][desc_col1],
                       f"{id_col2}": stories2.iloc[j][id_col2],
                       f"{desc_col2}": stories2.iloc[j][desc_col2],
                       "Similarity": round(score, 4)
                   })
       df_results = pd.DataFrame(matches)
       if not df_results.empty:
           st.success("✅ Matching complete!")
           st.dataframe(df_results)
           buffer = io.BytesIO()
           with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
               df_results.to_excel(writer, index=False, sheet_name="Results")
           buffer.seek(0)
           st.download_button(
               label="📥 Download Matching Results",
               data=buffer,
               file_name="Matching_Stories.xlsx",
               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
           )
       else:
           st.warning("⚠️ No similar user stories found with the current threshold.")
   except Exception as e:
       st.error(f"❌ Error processing files: {e}")
