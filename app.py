import streamlit as st
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.eda import EDAComponent
from src.components.model_trainer import ModelTrainer
from src.pipeline.predict_pipeline import PredictPipeline
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
eda = EDAComponent()
trainer = ModelTrainer()

# File upload
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xls", "xlsx"])

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

            if 'processed_df' in st.session_state:
                processed_df = st.session_state['processed_df']
                st.write("### Processed Dataset Preview")
                st.dataframe(processed_df.head())

                # Step 3.5: Exploratory Data Analysis (EDA)
                st.write("---")
                st.write("### 📊 Exploratory Data Analysis (on Processed Data)")
                eda_col = st.selectbox("Select column for EDA", processed_df.columns)
                if st.button("Run EDA"):
                    with st.spinner("Analyzing text..."):
                        stats = eda.get_corpus_stats(processed_df[eda_col])
                        freq_words = eda.get_frequent_words(processed_df[eda_col])
                        word_cloud_fig = eda.generate_word_cloud(processed_df[eda_col])

                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("#### Corpus Statistics")
                            st.json(stats)

                            st.write("#### Top 10 Frequent Words")
                            st.table(pd.DataFrame(freq_words, columns=["Word", "Count"]))

                        with col2:
                            st.write("#### Word Cloud (Top 10)")
                            if word_cloud_fig:
                                st.pyplot(word_cloud_fig)
                            else:
                                st.error("Failed to generate word cloud.")

                # Step 4: Prepare Dataset for Model
                st.write("---")
                st.write("### 🎯 Prepare Dataset for Model")
                
                target_col = st.selectbox("Select target/independent column", processed_df.columns)
                
                # Columns available for combining (exclude target)
                combine_options = [col for col in processed_df.columns if col != target_col]
                combine_cols = st.multiselect(
                    "Select columns to combine into text",
                    combine_options,
                    default=combine_options,
                    help="Selected columns will be combined into a single 'combined_text' column. All other columns will be dropped."
                )
                
                if st.button("Prepare for Model"):
                    if not combine_cols:
                        st.warning("Please select at least one column to combine.")
                    else:
                        with st.spinner("Preparing dataset..."):
                            model_df = transformation.prepare_for_model(processed_df, target_col, combine_cols)
                            st.session_state['model_df'] = model_df
                            st.success("Dataset prepared for model training!")
                            st.write("### Model-Ready Dataset Preview")
                            st.dataframe(model_df.head())
                            st.write(f"**Shape:** {model_df.shape}")
                            st.write(f"**Columns:** {model_df.columns.tolist()}")

                # Step 5: Train Model
                if 'model_df' in st.session_state:
                    st.write("---")
                    st.write("### 🤖 Train Model (TF-IDF + Naive Bayes)")

                    model_df = st.session_state['model_df']
                    text_col = st.selectbox("Select text column", model_df.columns, key="text_col")
                    target_model_col = st.selectbox("Select target column", model_df.columns, key="target_model_col")

                    # Show class distribution for the selected target
                    if target_model_col:
                        class_counts = model_df[target_model_col].value_counts()
                        st.write(f"**Class distribution for '{target_model_col}':** {len(class_counts)} classes")
                        st.dataframe(class_counts.reset_index().rename(
                            columns={"index": target_model_col, target_model_col: "Count"}
                        ).head(10))

                    if st.button("Train Model"):
                        if text_col == target_model_col:
                            st.error("Text column and target column cannot be the same!")
                        else:
                            with st.spinner("Training model..."):
                                # Split data
                                X_train, X_test, y_train, y_test = trainer.split_data(
                                    model_df, text_col, target_model_col, test_size=0.3
                                )

                                # Train model
                                pipeline = trainer.train_model()

                                # Get metrics
                                metrics = trainer.get_metrics()
                                st.session_state['metrics'] = metrics
                                st.session_state['pipeline'] = pipeline
                                st.session_state['trainer'] = trainer

                                st.success("Model trained successfully!")

                    # Display metrics
                    if 'metrics' in st.session_state:
                        metrics = st.session_state['metrics']

                        st.write("---")
                        st.write("### 📊 Model Metrics")

                        # Metrics in columns
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{metrics['accuracy']}%")
                        with col2:
                            st.metric("Precision", f"{metrics['precision']}%")
                        with col3:
                            st.metric("Recall", f"{metrics['recall']}%")
                        with col4:
                            st.metric("F1 Score", f"{metrics['f1_score']}%")

                        # Train/Test split info
                        st.info(f"**Train Size:** {metrics['train_size']} | **Test Size:** {metrics['test_size']} (70/30 split)")

                        # Confusion Matrix
                        st.write("#### Confusion Matrix")
                        cm_df = pd.DataFrame(
                            metrics['confusion_matrix'],
                            index=[f"Actual {i}" for i in range(len(metrics['confusion_matrix']))],
                            columns=[f"Predicted {i}" for i in range(len(metrics['confusion_matrix'][0]))]
                        )
                        st.dataframe(cm_df)

                        # Classification Report
                        st.write("#### Classification Report")
                        st.code(metrics['classification_report'])

                        # Prediction Section
                        st.write("---")
                        st.write("### 🔮 Test Prediction")

                        predict_pipeline = PredictPipeline(st.session_state['pipeline'])
                        user_input = st.text_area("Enter text to predict:")

                        if st.button("Predict"):
                            if user_input.strip():
                                prediction = predict_pipeline.predict(user_input)
                                st.success(f"**Prediction:** {prediction}")
                            else:
                                st.warning("Please enter some text to predict.")

        # Final Export
        if 'processed_df' in st.session_state:
            st.write("---")
            st.write("### Export Result")
            
            # Export options
            export_option = st.radio("Select export option:", ["Processed Dataset", "Model-Ready Dataset"], horizontal=True)
            
            if export_option == "Processed Dataset" and 'processed_df' in st.session_state:
                csv = st.session_state['processed_df'].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Processed CSV",
                    data=csv,
                    file_name="processed_nlp_dataset.csv",
                    mime="text/csv",
                )
            elif export_option == "Model-Ready Dataset" and 'model_df' in st.session_state:
                csv = st.session_state['model_df'].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Model-Ready CSV",
                    data=csv,
                    file_name="model_ready_dataset.csv",
                    mime="text/csv",
                )

    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.error(f"Application error: {str(e)}")

else:
    st.info("Please upload a CSV file to get started.")
