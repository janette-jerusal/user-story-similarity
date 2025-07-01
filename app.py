import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

# Page config
st.set_page_config(page_title="User Story Similarity Tool", layout="wide")
st.title("User Story Similarity Comparison Tool")
st.markdown("Upload **1 or 2 Excel files** with `ID` and `Desc` columns. The app compares user stories and identifies similar descriptions.")

uploaded_files = st.file_uploader("📤 Upload Excel File(s)", type=["xlsx"], accept_multiple_files=True)

def load_data(file, source_name):
    df = pd.read_excel(file)
    df.columns = [col.strip() for col in df.columns]
    df = df.rename(columns={df.columns[0]: 'ID', df.columns[1]: 'Desc'})
    df['Source'] = source_name
    return df

def compute_similarity(df1, df2, threshold):
    combined_desc = pd.concat([df1['Desc'], df2['Desc']], ignore_index=True)
    tfidf = TfidfVectorizer().fit_transform(combined_desc)
    tfidf_df1 = tfidf[:len(df1)]
    tfidf_df2 = tfidf[len(df1):]
    similarity_matrix = cosine_similarity(tfidf_df1, tfidf_df2)

    results = []
    for i, row in enumerate(similarity_matrix):
        for j, score in enumerate(row):
            if score >= threshold:
                results.append({
                    'Story A ID': df1.iloc[i]['ID'],
                    'Story A Desc': df1.iloc[i]['Desc'],
                    'Story B ID': df2.iloc[j]['ID'],
                    'Story B Desc': df2.iloc[j]['Desc'],
                    'Similarity Score': round(score, 3)
                })
    return pd.DataFrame(results)

def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Top Matches")
    return output.getvalue()

if uploaded_files:
    # Add threshold slider only when files are uploaded
    similarity_threshold = st.slider(
        "🔧 Set Similarity Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.75, 
        step=0.01
    )

    if len(uploaded_files) == 1:
        df = load_data(uploaded_files[0], uploaded_files[0].name)
        df1, df2 = df.copy(), df.copy()
    elif len(uploaded_files) == 2:
        df1 = load_data(uploaded_files[0], uploaded_files[0].name)
        df2 = load_data(uploaded_files[1], uploaded_files[1].name)
    else:
        st.error("Please upload only 1 or 2 Excel files.")
        st.stop()

    result_df = compute_similarity(df1, df2, similarity_threshold)

    # Display KPIs
    col1, col2 = st.columns(2)
    col1.metric("🧾 Total User Stories", len(df1) + len(df2))
    col2.metric("🔍 Matched Pairs", len(result_df))

    # Show result table
    st.subheader("📊 Top Matching User Stories")
    st.dataframe(result_df, use_container_width=True)

    # Download button
    excel_data = convert_df_to_excel(result_df)
    st.download_button(
        label="Download Results as Excel",
        data=excel_data,
        file_name="user_story_matches.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("⬆️ Upload one or two Excel files to begin.")

 
