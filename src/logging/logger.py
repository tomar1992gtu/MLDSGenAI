import logging
import os

'''
Logs generation in "2026-05-28 12:45:10,123 - INFO - Application started" format
'''

LOG_DIR = "artifacts/logs"
os.makedirs(LOG_DIR, exist_ok=True)     # exist_ok=True means: Don’t throw an error if the folder already exists

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "running.log"),
    level=logging.INFO,                                     # Level can be "INFO, WARNING, DEBUG, ERROR"
    format="%(asctime)s - %(levelname)s - %(message)s"      # %(asctime)s → timestamp,
                                                            # %(levelname)s → log level (INFO, ERROR, etc.),
                                                            # %(message)s → actual log message
)

logger = logging.getLogger()
