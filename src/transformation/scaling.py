from datetime import datetime

from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler)

import pandas as pd
import joblib
import os
from src.constants.global_constants import ARTIFACTS_ENCODER_DIR
from src.logging.logger import logger


class DataScaling:

    scaler = None

    #################################################
    # FIT + TRANSFORM
    #################################################
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
        try:
            logger.info(f"Scaling Started | Scaler: {scaler_name}")
            scaler_name = scaler_name.lower()

            if scaler_name == "standard":
                cls.scaler = StandardScaler()

            elif scaler_name == "minmax":
                cls.scaler = MinMaxScaler()

            elif scaler_name == "robust":
                cls.scaler = RobustScaler()

            else:
                logger.error(f"Unsupported scaler: {scaler_name}")
                raise ValueError(
                    "Unsupported scaler. Choose from "
                    "'standard', 'minmax', or 'robust'."
                )

            logger.info(f"Training Shape Before Scaling: {X_train.shape}")
            # Fit and transform training data
            X_train_scaled = pd.DataFrame(
                            cls.scaler.fit_transform(X_train),
                            columns=X_train.columns,
                            index=X_train.index
                        )
            logger.info("Scaler fitted successfully")

            # Create directory if it doesn't exist
            os.makedirs(ARTIFACTS_ENCODER_DIR, exist_ok=True)

            # Create timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            scaler_path = os.path.join(ARTIFACTS_ENCODER_DIR, f"{model_name}_scaler_{timestamp}.pkl")

            # Save scaler
            joblib.dump(cls.scaler, scaler_path)
            logger.info(f"Scaler Saved: {scaler_path}")
            logger.info("Scaling Completed Successfully")

            return X_train_scaled, scaler_path

        except Exception:
            logger.exception("Scaling Failed")
            raise

    #################################################
    # TRANSFORM
    #################################################
    @classmethod
    def transform(cls, X_data):
        try:
            """
            Transform data using the fitted/loaded scaler.
            """
            if cls.scaler is None:
                logger.error("Scaler has not been loaded.")
                raise ValueError("Scaler has not been fitted or loaded yet.")

            logger.info(f"Applying scaler on shape: {X_data.shape}")
            transformed = cls.scaler.transform(X_data)
            logger.info("Scaling Transformation Completed")
            return pd.DataFrame(
                    cls.scaler.transform(X_data),
                    columns=X_data.columns,
                    index=X_data.index
                )

        except Exception:
            logger.exception("Scaler Transformation Failed")
            raise

    #################################################
    # LOAD SCALER
    #################################################
    @classmethod
    def load_latest_scaler(cls, model_name):
        try:
            files = [f for f in os.listdir(ARTIFACTS_ENCODER_DIR)
                    if f.startswith(f"{model_name}_scaler") and f.endswith(".pkl")]

            if not files:
                logger.error(f"No scaler found for model: {model_name}")
                raise FileNotFoundError(f"No scaler found for {model_name}")

            latest = sorted(files)[-1]
            scaler_path = os.path.join(ARTIFACTS_ENCODER_DIR, latest)
            logger.info(f"Loading Scaler: {scaler_path}")
            cls.scaler = joblib.load(scaler_path)
            logger.info("Scaler Loaded Successfully")
            return cls.scaler

        except Exception:
            logger.exception("Scaler Loading Failed")
            raise
