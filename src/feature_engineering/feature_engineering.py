import os
from datetime import datetime

import joblib
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
from src.constants.global_constants import ENCODER_DIR
from src.logging.logger import logger


class FeatureEngineering:

    def __init__(self):
        self.onehot_encoder = None
        self.label_encoder = None
        self.categorical_cols = None

    # -------------------------
    # 1. Encode INPUT FEATURES
    # -------------------------
    def fit_features(self, df: pd.DataFrame, categorical_cols, model_name="model"):
        try:
            logger.info("Feature Engineering Started")
            df = df.copy()

            if not categorical_cols:
                logger.info("No categorical columns found. Skipping feature encoding.")
                return df

            logger.info(f"Categorical Columns: {categorical_cols}")

            # Initialize encoder
            self.onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

            # Fit + transform categorical
            encoded = self.onehot_encoder.fit_transform(df[categorical_cols])

            # Convert to DataFrame
            encoded_df = pd.DataFrame(
                encoded,
                columns=self.onehot_encoder.get_feature_names_out(categorical_cols),
                index=df.index
            )
            logger.info(f"Generated {len(encoded_df.columns)} encoded columns")

            # Save fitted feature encoder
            os.makedirs(ENCODER_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            encoder_path = f"{ENCODER_DIR}/{model_name}_feature_encoder_{timestamp}.pkl"
            joblib.dump(self.onehot_encoder, encoder_path)
            logger.info(f"Feature encoder saved: {encoder_path}")

            # Drop original categorical columns
            df = df.drop(columns=categorical_cols)

            # Merge encoded columns
            df = pd.concat([df, encoded_df], axis=1)
            logger.info(f"Feature Engineering Completed. Shape: {df.shape}")
            return df

        except Exception:
            logger.exception("Feature Encoding Failed")
            raise

    # -------------------------
    # 2. Encode TARGET COLUMN
    # -------------------------
    def encode_target(self, df: pd.DataFrame, target_col: str, model_name="model"):
        try:
            df = df.copy()
            #target_col = target_col.lower().strip()

            logger.info(f"Target Encoding Started: {target_col}")
            self.label_encoder = LabelEncoder()
            df[target_col] = self.label_encoder.fit_transform(df[target_col])

            # Save fitted target encoder
            os.makedirs(ENCODER_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            encoder_path = f"{ENCODER_DIR}/{model_name}_target_encoder_{timestamp}.pkl"
            joblib.dump(self.label_encoder, encoder_path)
            logger.info(f"Target encoder saved: {encoder_path}")
            logger.info(f"Target Classes: {list(self.label_encoder.classes_)}")

            return df

        except Exception:
            logger.exception("Target Encoding Failed")
            raise



    ##################################################
    # LOAD FEATURE ENCODER
    ##################################################
    def load_latest_feature_encoder(self, model_name="model"):
        files = [f for f in os.listdir(ENCODER_DIR)
                if f.startswith(f"{model_name}_feature_encoder") and f.endswith(".pkl")]

        if not files:
            logger.error(f"No feature encoder found for {model_name}")
            raise FileNotFoundError(f"No feature encoder found for {model_name}")

        latest = sorted(files)[-1]
        encoder_path = os.path.join(ENCODER_DIR, latest)
        logger.info(f"Loading Feature Encoder: {encoder_path}")

        return joblib.load(os.path.join(ENCODER_DIR, latest))

    ##################################################
    # TRANSFORM FEATURES
    ##################################################
    def transform_features(self, df, categorical_cols, model_name="model"):
        try:
            if not categorical_cols:
                logger.info("No categorical columns found. Skipping transform.")
                return df

            logger.info(f"Transforming categorical columns: {categorical_cols}")
            encoder = self.load_latest_feature_encoder(model_name)
            encoded = encoder.transform(df[categorical_cols])
            encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)

            df = df.drop(columns=categorical_cols)
            df = pd.concat([df, encoded_df], axis=1)
            logger.info(f"Inference Feature Engineering Completed. Shape: {df.shape}")

            return df
        except Exception:
            logger.exception("Feature Transformation Failed")
            raise
