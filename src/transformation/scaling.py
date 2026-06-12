from datetime import datetime

from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler)

import joblib
import os
from src.constants.global_constants import ARTIFACTS_ENCODER_DIR

class DataScaling:

    scaler = None

    @classmethod
    def fit_transform(cls, X_train, model_name, scaler_name="standard"):
        """
        Fit the selected scaler on training data,
        transform the data, and save the scaler.

        Parameters
        ----------
        X_train : array-like or DataFrame
            Training features.
        model_name : str
            Model name used in scaler filename.
        scaler_name : str, default="standard"
            One of: "standard", "minmax", "robust".

        Returns
        -------
        tuple
            (X_train_scaled, scaler_path)
        """

        scaler_name = scaler_name.lower()

        if scaler_name == "standard":
            cls.scaler = StandardScaler()

        elif scaler_name == "minmax":
            cls.scaler = MinMaxScaler()

        elif scaler_name == "robust":
            cls.scaler = RobustScaler()

        else:
            raise ValueError(
                "Unsupported scaler. Choose from "
                "'standard', 'minmax', or 'robust'."
            )

        # Fit and transform training data
        X_train_scaled = cls.scaler.fit_transform(X_train)

        # Create directory if it doesn't exist
        os.makedirs(ARTIFACTS_ENCODER_DIR, exist_ok=True)

        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scaler_path = os.path.join(ARTIFACTS_ENCODER_DIR, f"{model_name}_scaler_{timestamp}.pkl")

        # Save scaler
        joblib.dump(cls.scaler, scaler_path)

        return X_train_scaled, scaler_path

    @classmethod
    def transform(cls, X_data):
        """
        Transform data using the fitted/loaded scaler.
        """
        if cls.scaler is None:
            raise ValueError("Scaler has not been fitted or loaded yet.")

        return cls.scaler.transform(X_data)

    @classmethod
    def load_latest_scaler(cls, model_name):
        files = [f for f in os.listdir(ARTIFACTS_ENCODER_DIR)
                if f.startswith(f"{model_name}_scaler") and f.endswith(".pkl")]

        latest = sorted(files)[-1]
        path = os.path.join(ARTIFACTS_ENCODER_DIR, latest)
        cls.scaler = joblib.load(path)

        return cls.scaler
