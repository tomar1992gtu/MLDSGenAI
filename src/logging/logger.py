import logging
import os
from datetime import datetime

from src.constants.global_constants import ARTIFACTS_LOGS_DIR

'''
Logs generation in "2026-05-28 12:45:10,123 - INFO - Application started" format
'''

os.makedirs(ARTIFACTS_LOGS_DIR, exist_ok=True)     # exist_ok=True means: Don’t throw an error if the folder already exists
# Generate timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

log_file = os.path.join(ARTIFACTS_LOGS_DIR, f"running_{timestamp}.log")
logging.basicConfig(
    filename=os.path.join(ARTIFACTS_LOGS_DIR, log_file),
    level=logging.INFO,                                     # Level can be "INFO, WARNING, DEBUG, ERROR"
    format="%(asctime)s - %(levelname)s - %(message)s"      # %(asctime)s → timestamp,
                                                            # %(levelname)s → log level (INFO, ERROR, etc.),
                                                            # %(message)s → actual log message
)

logger = logging.getLogger()
