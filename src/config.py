from loguru import logger
import sys
import os
from pathlib import Path

class Config:
    ALLOWED_FILE_EXTENSIONS = set([".pdf", ".md", ".txt"])
    SEED = 42

    class Model:
        NAME = "deepseek-r1:14b"
        TEMPERATURE = 0.6

    class Preprocessing:
        CHUNK_SIZE = 2048
        CHUNK_OVERLAP = 128
        EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
        RERANKER = "ms-marco-MiniLM-L-12-v2"
        LLM = "llama3.2"
        CONTEXTUALIZE_CHUNKS = True
        N_SEMANTIC_RESULTS = 5
        N_BM25_RESULTS = 5

    class Chatbot:
        N_CONTEXT_RESULTS = 3

    class Path:
        APP_HOME = Path(os.getenv("APP_HOME", Path(__file__).parent.parent))
        DATA_DIR = APP_HOME / "data"


def configure_logging():
    config = {
        "handlers": [
            {
                "sink": sys.stdout, 
                "colorize": True,
                "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
            },
            {
                "sink": "app.log",  # Log to a file as well
                "rotation": "10 MB",  # Rotate log file when it reaches 10MB
                "compression": "zip", # Compress old log files
                "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            },
        ],
        "levels": [
            {"name": "TRACE", "color": "<cyan>"},
            {"name": "DEBUG", "color": "<blue>"},
            {"name": "INFO", "color": "<green>"},
            {"name": "WARNING", "color": "<yellow>"},
            {"name": "ERROR", "color": "<red>"},
            {"name": "CRITICAL", "color": "<bold><red>"},
        ],
    }

    logger.configure(**config)

# Example usage:
configure_logging()

logger.trace("This is a trace message.")
logger.debug("This is a debug message.")
logger.info("This is an info message.")
logger.warning("This is a warning message.")
logger.error("This is an error message.")
logger.critical("This is a critical message.")