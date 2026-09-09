import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger()

# Download necessary NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

stop_words = set(stopwords.words('english'))
stem = PorterStemmer()

class DataTransformation:
    def __init__(self):
        pass

    def preprocess_text(self, text):
        """
        Basic NLP preprocessing for a single piece of text.
        """
        if not isinstance(text, str):
            return ""

        # 1. Lowercase
        text = text.lower()

        # Mark URLs, symbols, and digits instead of deleting (as requested)
        text = re.sub(r'https?://\S+|www\.\S+', 'URL_TOKEN', text)
        text = re.sub(r'[^\w\s]', 'SYM_TOKEN', text)
        text = re.sub(r'\d+', 'NUM_TOKEN', text)

        # Note: The user provided redundant removal steps in their prompt.
        # Since they specifically mentioned "Mark URLs instead of deleting",
        # I'm omitting the subsequent re.sub calls that would delete them.

        # 3. Remove special quotes and HTML tags (Still useful for cleanup)
        text = re.sub(r"[‘’“”«»]", "", text)
        text = re.sub(r'<.*?>', '', text)

        # 5. Remove remaining punctuation (if any left after SYM_TOKEN)
        text = text.translate(str.maketrans('', '', string.punctuation))

        # 6. Tokenize
        words = word_tokenize(text)

        # 7. Remove stopwords & Stemming
        words = [stem.stem(word) for word in words if word not in stop_words]

        return " ".join(words)

    def clean_dataset(self, df):
        """
        Basic cleaning: removing null values and duplicated values.
        """
        try:
            logger.info("Starting basic dataset cleaning...")
            initial_shape = df.shape

            # Remove duplicates
            df = df.drop_duplicates()
            logger.info(f"Removed duplicates. Shape changed from {initial_shape} to {df.shape}")

            # Remove null values
            df = df.dropna()
            logger.info(f"Removed null values. Shape changed from {df.shape} to {df.shape}")

            return df
        except Exception as e:
            logger.error(f"Error during basic cleaning: {str(e)}")
            raise e

    def transform_text_columns(self, df, columns):
        """
        Applies preprocess_text to specified columns in the DataFrame.
        """
        try:
            logger.info(f"Applying NLP preprocessing to columns: {columns}")
            for col in columns:
                if col in df.columns:
                    df[col] = df[col].apply(self.preprocess_text)
                    logger.info(f"Finished preprocessing column: {col}")
                else:
                    logger.warning(f"Column {col} not found in dataset.")
            return df
        except Exception as e:
            logger.error(f"Error during text transformation: {str(e)}")
            raise e

    def prepare_for_model(self, df, target_column):
        """
        Prepares dataset for model training:
        1. User selects target column (independent variable)
        2. Creates combined_text column from all remaining non-numerical columns
        3. Drops other columns, keeping only target and combined_text
        
        Args:
            df: Input DataFrame
            target_column: Column name to use as target/independent variable
            
        Returns:
            DataFrame with target_column and combined_text columns
        """
        try:
            logger.info(f"Preparing dataset for model. Target column: {target_column}")
            
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset")
            
            # Get numerical columns to exclude
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # Get text columns (all columns except target and numerical)
            text_cols = [col for col in df.columns if col != target_column and col not in numerical_cols]
            
            logger.info(f"Text columns to combine: {text_cols}")
            logger.info(f"Numerical columns (will be dropped): {numerical_cols}")
            
            # Create combined text column
            df_result = df.copy()
            df_result['combined_text'] = df_result[text_cols].apply(
                lambda row: ' '.join(row.dropna().astype(str)), axis=1
            )
            
            # Keep only target and combined_text
            df_result = df_result[[target_column, 'combined_text']]
            
            logger.info(f"Dataset prepared. Final shape: {df_result.shape}")
            return df_result
            
        except Exception as e:
            logger.error(f"Error preparing dataset for model: {str(e)}")
            raise e
