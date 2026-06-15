import joblib
import os
from datetime import datetime

from src.constants.global_constants import TRAINED_MODELS_DIR
from src.logging.logger import logger


class TrainModel:

    @classmethod
    def train(cls, model, X_train, y_train, model_name="model"):

        try:
            logger.info(f"Model Training Started: {model_name}")
            logger.info(f"Training Features Shape: {X_train.shape}")
            logger.info(f"Training Target Shape: {y_train.shape}")
            # Train model
            model.fit(X_train, y_train)

            logger.info(f"Training Completed Successfully: {model_name}")
            # Ensure directory exists
            os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)

            # Add timestamp for versioning
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            ''''
            # Models are saved in .pkl (pickle) format because Python uses Pickle serialization to store objects in a file so they can be loaded later exactly as they were.
            # It stores a Python object in binary format, such as:
                # trained ML models
                # dictionaries
                # lists
                # pipelines
                # encoders
            '''
            model_path = f"{TRAINED_MODELS_DIR}/{model_name}_{timestamp}.pkl"

            # Save model
            joblib.dump(model, model_path)
            logger.info(f"Model Saved Successfully: {model_path}")
            return model, model_path

        except Exception:
            logger.exception(f"Model Training Failed: {model_name}")
            raise
