import pandas as pd
from src.utils.logger import get_logger

logger = get_logger()


class PredictPipeline:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict(self, text):
        """
        Predicts the class for a single text input.
        """
        try:
            logger.info(f"Predicting for input: {text[:50]}...")
            prediction = self.pipeline.predict([text])
            return prediction[0]
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise e

    def predict_batch(self, texts):
        """
        Predicts classes for a list of texts.
        """
        try:
            logger.info(f"Batch predicting for {len(texts)} inputs...")
            predictions = self.pipeline.predict(texts)
            return predictions.tolist()
        except Exception as e:
            logger.error(f"Error during batch prediction: {str(e)}")
            raise e
