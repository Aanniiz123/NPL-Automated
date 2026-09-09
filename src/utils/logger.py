import logging
import os
from datetime import datetime

LOG_FILE = "logs/nlp_process.log"

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger("NLP_Process")

def get_logger():
    return logger
