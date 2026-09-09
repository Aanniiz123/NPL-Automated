import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from src.utils.logger import get_logger

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

logger = get_logger()

class EDAComponent:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def _preprocess_text(self, text):
        """Internal helper to tokenize and remove stopwords."""
        if not isinstance(text, str):
            return []
        tokens = word_tokenize(text.lower())
        # Keep only alphanumeric tokens and remove stopwords
        return [word for word in tokens if word.isalnum() and word not in self.stop_words]

    def get_corpus_stats(self, text_series):
        """Calculates basic corpus statistics."""
        try:
            logger.info("Calculating corpus statistics...")
            all_words = []
            for text in text_series:
                all_words.extend(self._preprocess_text(text))

            total_words = len(all_words)
            unique_words = len(set(all_words))
            avg_word_length = sum(len(word) for word in all_words) / total_words if total_words > 0 else 0

            return {
                "Total Words": total_words,
                "Unique Words": unique_words,
                "Average Word Length": round(avg_word_length, 2)
            }
        except Exception as e:
            logger.error(f"Error calculating corpus stats: {str(e)}")
            return None

    def get_frequent_words(self, text_series, top_n=10):
        """Identifies the most frequent words."""
        try:
            logger.info(f"Identifying top {top_n} frequent words...")
            all_words = []
            for text in text_series:
                all_words.extend(self._preprocess_text(text))

            counts = Counter(all_words)
            return counts.most_common(top_n)
        except Exception as e:
            logger.error(f"Error identifying frequent words: {str(e)}")
            return []

    def generate_word_cloud(self, text_series):
        """Generates a word cloud for the given text series."""
        try:
            logger.info("Generating word cloud...")
            text_combined = " ".join(text_series.astype(str))

            # Generate word cloud
            wordcloud = WordCloud(width=800, height=400,
                                  background_color='white',
                                  max_words=10).generate(text_combined)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            plt.tight_layout()

            return fig
        except Exception as e:
            logger.error(f"Error generating word cloud: {str(e)}")
            return None
