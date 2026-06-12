import os
import joblib
import pandas as pd

from src.transformation.scaling import DataScaling
from src.constants.global_constants import TRAINED_MODELS_DIR


class PredictionPipeline:

    model = None

    @classmethod
    def load_model(cls, model_name):
        files = [
            f for f in os.listdir(TRAINED_MODELS_DIR)
            if f.startswith(model_name)
            and f.endswith(".pkl")
        ]

        if not files:
            raise FileNotFoundError(f"No model found for {model_name}")

        latest = sorted(files)[-1]
        model_path = os.path.join(TRAINED_MODELS_DIR, latest)
        cls.model = joblib.load(model_path)

        return cls.model

    @classmethod
    def predict(cls, data, model_name):
        # Apply scaling if exists
        '''if scaler_path:
            DataScaling.load_scaler(scaler_path)
            data = DataScaling.transform(data)'''

        model = cls.load_model(model_name)
        prediction = model.predict(data)
        result = {"prediction": prediction.tolist()}

        if hasattr(model, "predict_proba"):
            result["probability"] = model.predict_proba(data).tolist()

        return result

