import streamlit as st
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger()

st.set_page_config(page_title="Automated NLP Process", layout="wide")

st.title("🤖 Automated NLP Data Cleaning Pipeline")
st.markdown("""
This app automates the first step of an NLP pipeline:
**Data Ingestion $\rightarrow$ Basic Cleaning $\rightarrow$ Text Preprocessing**.
""")

# Initialize components
ingestion = DataIngestion()
transformation = DataTransformation()

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Step 1: Data Ingestion
        with st.spinner("Ingesting data..."):
            df = ingestion.load_dataset(uploaded_file)
            st.success("Dataset loaded successfully!")
            st.write("### Original Dataset Preview")
            st.dataframe(df.head())

        # Step 2: Basic Cleaning
        if st.button("Run Basic Cleaning (Nulls & Duplicates)"):
            with st.spinner("Cleaning..."):
                cleaned_df = transformation.clean_dataset(df)
                st.session_state['cleaned_df'] = cleaned_df
                st.success("Basic cleaning complete!")
                st.write("### Cleaned Dataset Preview")
                st.dataframe(cleaned_df.head())

        # Step 3: NLP Preprocessing
        if 'cleaned_df' in st.session_state:
            df_to_process = st.session_state['cleaned_df']

            st.write("---")
            st.write("### NLP Text Preprocessing")

            # Select columns to preprocess
            all_columns = df_to_process.columns.tolist()
            selected_cols = st.multiselect("Select columns for NLP preprocessing", all_columns)

            if st.button("Apply NLP Preprocessing"):
                if not selected_cols:
                    st.warning("Please select at least one column to process.")
                else:
                    with st.spinner("Processing text..."):
                        processed_df = transformation.transform_text_columns(df_to_process, selected_cols)
                        st.session_state['processed_df'] = processed_df
                        st.success("NLP preprocessing complete!")
                        st.write("### Processed Dataset Preview")
                        st.dataframe(processed_df.head())

        # Final Export
        if 'processed_df' in st.session_state:
            st.write("---")
            st.write("### Export Result")
            csv = st.session_state['processed_df'].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Processed CSV",
                data=csv,
                file_name="processed_nlp_dataset.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.error(f"Application error: {str(e)}")

else:
    st.info("Please upload a CSV file to get started.")
