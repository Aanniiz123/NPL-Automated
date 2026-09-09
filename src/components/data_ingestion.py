import pandas as pd
from src.utils.logger import get_logger

logger = get_logger()


class DataIngestion:
    def __init__(self):
        pass

    def load_dataset(self, file):
        """
        Loads a CSV or Excel file into a pandas DataFrame.
        :param file: File object or path to the file.
        :return: pandas DataFrame.
        """
        try:
            logger.info("Starting data ingestion...")
            file_name = file.name.lower()
            file_ext = file_name.split('.')[-1]

            if file_ext == 'csv':
                df = pd.read_csv(file)
            elif file_ext in ['xls', 'xlsx']:
                df = pd.read_excel(file)
            else:
                raise ValueError(f"Unsupported file format: .{file_ext}. Please upload a CSV, XLS, or XLSX file.")

            logger.info(f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
            return df
        except Exception as e:
            logger.error(f"Error during data ingestion: {str(e)}")
            raise e