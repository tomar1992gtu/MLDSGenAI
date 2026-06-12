import os
from datetime import datetime

import joblib
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
from src.constants.global_constants import ENCODER_DIR

class FeatureEngineering:

    def __init__(self):
        self.onehot_encoder = None
        self.label_encoder = None
        self.categorical_cols = None

    # -------------------------
    # 1. Encode INPUT FEATURES
    # -------------------------
    def fit_features(self, df: pd.DataFrame, categorical_cols, model_name="model"):

        df = df.copy()
        #self.categorical_cols = [col.lower().strip() for col in categorical_cols]
        #categorical_cols = self.categorical_cols
        #print("DF Columns: {df.columns.tolist())}")
        #print("Categorical Columns: {categorical_cols}")

        if not categorical_cols:
            print("[INFO] No categorical columns found. Skipping encoding.")
            return df

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

        # Save fitted feature encoder
        os.makedirs(ENCODER_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        encoder_path = f"{ENCODER_DIR}/{model_name}_feature_encoder_{timestamp}.pkl"
        joblib.dump(self.onehot_encoder, encoder_path)

        # Drop original categorical columns
        df = df.drop(columns=categorical_cols)

        # Merge encoded columns
        df = pd.concat([df, encoded_df], axis=1)

        return df

    # -------------------------
    # 2. Encode TARGET COLUMN
    # -------------------------
    def encode_target(self, df: pd.DataFrame, target_col: str, model_name="model"):

        df = df.copy()
        #target_col = target_col.lower().strip()

        self.label_encoder = LabelEncoder()
        df[target_col] = self.label_encoder.fit_transform(df[target_col])

        # Save fitted target encoder
        os.makedirs(ENCODER_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        encoder_path = f"{ENCODER_DIR}/{model_name}_target_encoder_{timestamp}.pkl"
        joblib.dump(self.label_encoder, encoder_path)

        return df



    # Training and inference must use SAME encoder.
    def load_latest_feature_encoder(self, model_name="model"):
        files = [f for f in os.listdir(ENCODER_DIR)
                if f.startswith(f"{model_name}_feature_encoder") and f.endswith(".pkl")]

        if not files:
            raise FileNotFoundError("No feature encoder found")

        latest = sorted(files)[-1]
        return joblib.load(os.path.join(ENCODER_DIR, latest))

    def transform_features(self, df, categorical_cols, model_name="model"):
        if not categorical_cols:
            return df
        encoder = self.load_latest_feature_encoder(model_name)
        encoded = encoder.transform(df[categorical_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)

        df = df.drop(columns=categorical_cols)
        df = pd.concat([df, encoded_df], axis=1)

        return df
