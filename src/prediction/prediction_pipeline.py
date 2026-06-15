import os
import joblib
import pandas as pd

from src.logging.logger import logger
from src.transformation.scaling import DataScaling
from src.constants.global_constants import TRAINED_MODELS_DIR


class PredictionPipeline:

    model = None

    @classmethod
    def load_model(cls, model_name):
        try:
            logger.info(f"Searching model: {model_name}")
            files = [f for f in os.listdir(TRAINED_MODELS_DIR)
                if f.startswith(model_name)
                and f.endswith(".pkl")]

            if not files:
                logger.error(f"No model found for {model_name}")
                raise FileNotFoundError(f"No model found for {model_name}")

            latest = sorted(files)[-1]
            model_path = os.path.join(TRAINED_MODELS_DIR, latest)
            logger.info(f"Loading model from {model_path}")
            cls.model = joblib.load(model_path)
            logger.info("Model loaded successfully")
            return cls.model

        except Exception as e:
            logger.exception(f"Failed loading model {model_name}: {str(e)}")
            raise

    @classmethod
    def predict(cls, data, model_name):
        try:
            # Apply scaling if exists
            '''if scaler_path:
                DataScaling.load_scaler(scaler_path)
                data = DataScaling.transform(data)'''

            logger.info(f"Prediction started for model: {model_name}")

            if isinstance(data, pd.DataFrame):
                logger.info(f"Input shape: {data.shape}")

            model = cls.load_model(model_name)
            prediction = model.predict(data)
            logger.info(f"Prediction generated: {prediction.tolist()}")
            result = { "prediction": prediction.tolist()}

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(data)
                logger.info("Prediction probabilities generated")
                result["probability"] = (probabilities.tolist())

            logger.info("Prediction completed successfully")
            return result

        except Exception as e:
            logger.exception(f"Prediction failed for model {model_name}: {str(e)}")
            raise
