import streamlit as st
import pandas as pd
import google.generativeai as genai
import streamlit.components.v1 as components
import pandas_profiling

def setup_generative_ai():
    api_key = st.secrets["api_key"]
    genai.configure(api_key=api_key)

def upload_csv():
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df)
        return df
    return None

def generate_report(df):
    st.header("Dataset Summary")
    st.write(df.describe())
    st.write("Number of missing values:", df.isnull().sum().sum())

def ask_ai_question(df, question):
    response = genai.GenerativeModel('gemini-1.5-flash').generate_content(
        f"Based on this dataset: {df.head(5)}\nAnswer the following question:\n{question}")
    return response.text

def interactive_qna(df):
    st.header("Ask Questions About the Dataset")
    user_question = st.text_input("Enter your question:")
    if st.button("Ask"):
        if user_question:
            response = ask_ai_question(df, user_question)
            st.code(f"AI Response: {response}")

def perform_eda(df):
    st.header("Exploratory Data Analysis (EDA)")
    
    st.subheader("Missing Data")
    st.write(df.isnull().sum())

    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if not numeric_df.empty:
        corr = numeric_df.corr()
        st.write(corr.style.background_gradient(cmap='coolwarm'))
    else:
        st.warning("No numeric columns available for correlation.")

def detect_anomalies(df):
    st.header("Anomaly Detection")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    if not numeric_df.empty:
        z_scores = (numeric_df - numeric_df.mean()) / numeric_df.std()
        outliers = z_scores[(z_scores > 3) | (z_scores < -3)]
        outliers_cleaned = outliers.dropna(how='all')

        if not outliers_cleaned.empty:
            st.write("Outliers detected:")
            st.dataframe(outliers_cleaned)
        else:
            st.write("No significant outliers detected.")
    else:
        st.warning("No numeric columns available for anomaly detection.")

def auto_eda_report(df):
    profile = ProfileReport(df, title="Pandas Profiling Report")
    st.header("Auto EDA Report")
    st_profile_html = profile.to_html()
    components.html(st_profile_html, height=1000, scrolling=True)


def main():
    setup_generative_ai()
    st.set_page_config(page_title="Awesome LLM-Streamlit App", page_icon="📝")
    st.title("Interactive LLM-Data Assistant")
    df = upload_csv()
    
    if df is not None:
        generate_report(df)
        interactive_qna(df)
        perform_eda(df)
        detect_anomalies(df)
        auto_eda_report(df)

if __name__ == "__main__":
    main()
