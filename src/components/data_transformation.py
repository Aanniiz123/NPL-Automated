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
