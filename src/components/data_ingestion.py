import pandas as pd
from src.utils.logger import get_logger

logger = get_logger()

class DataIngestion:
    def __init__(self):
        pass

    def load_dataset(self, file):
        """
        Loads a CSV file into a pandas DataFrame.
        :param file: File object or path to the CSV file.
        :return: pandas DataFrame.
        """
        try:
            logger.info("Starting data ingestion...")
            df = pd.read_csv(file)
            logger.info(f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
            return df
        except Exception as e:
            logger.error(f"Error during data ingestion: {str(e)}")
            raise e
