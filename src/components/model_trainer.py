from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from src.utils.logger import get_logger

logger = get_logger()


class ModelTrainer:
    def __init__(self):
        self.pipeline = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

    def split_data(self, df, text_col, target_col, test_size=0.3, random_state=42):
        """
        Splits the data into train and test sets (70/30).
        """
        try:
            logger.info(f"Splitting data: {test_size*100}% test, {(1-test_size)*100}% train")

            X = df[text_col]
            y = df[target_col]

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            logger.info(f"Train size: {len(self.X_train)}, Test size: {len(self.X_test)}")
            return self.X_train, self.X_test, self.y_train, self.y_test

        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise e

    def train_model(self):
        """
        Builds and trains TF-IDF + Naive Bayes pipeline.
        """
        try:
            logger.info("Building TF-IDF + Naive Bayes pipeline...")

            self.pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    stop_words='english'
                )),
                ('classifier', MultinomialNB(alpha=1.0))
            ])

            logger.info("Training model...")
            self.pipeline.fit(self.X_train, self.y_train)

            self.y_pred = self.pipeline.predict(self.X_test)

            logger.info("Model training complete")
            return self.pipeline

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise e

    def get_metrics(self):
        """
        Calculates and returns all metrics.
        """
        try:
            logger.info("Calculating metrics...")

            accuracy = accuracy_score(self.y_test, self.y_pred)
            precision = precision_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(self.y_test, self.y_pred)
            report = classification_report(self.y_test, self.y_pred, zero_division=0)

            metrics = {
                "accuracy": round(accuracy * 100, 2),
                "precision": round(precision * 100, 2),
                "recall": round(recall * 100, 2),
                "f1_score": round(f1 * 100, 2),
                "confusion_matrix": cm.tolist(),
                "classification_report": report,
                "train_size": len(self.X_train),
                "test_size": len(self.X_test)
            }

            logger.info(f"Metrics - Accuracy: {metrics['accuracy']}%, F1: {metrics['f1_score']}%")
            return metrics

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise e
