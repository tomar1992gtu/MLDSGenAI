import os

import joblib
import pandas as pd
import yaml

from src.logging.logger import logger
from src.preprocessing.data_preprocessing import DataPreprocessing
from src.feature_engineering.feature_engineering import FeatureEngineering
from src.transformation.scaling import DataScaling
from src.prediction.prediction_pipeline import PredictionPipeline
from src.constants.global_constants import CONFIG_PATH, FEATURE_COLUMNS_DIR

class InferencePipeline:

    def __init__(self, model_name, categorical_columns=None):
        self.model_name = model_name
        #categorical_columns = [col.lower() for col in config.get("feature_engineering", {}).get("categorical_columns", [])]
        self.categorical_columns = [col.lower() for col in (categorical_columns or [])]
        logger.info(f"InferencePipeline Initialized | Model={model_name}")

    def run(self, input_data):
        try:
            logger.info("=====>>> Starting Inference Pipeline <<======")
            #################################
            # STEP 1 Convert input
            #################################
            logger.info("STEP 1 - Input Conversion Started")
            if not isinstance(input_data, pd.DataFrame):
                df = pd.DataFrame([input_data])
            else:
                df = input_data.copy()
            logger.info(f"Input Shape: {df.shape}")

            #################################
            # STEP 2 Preprocessing
            #################################
            logger.info("STEP 2 - Data Preprocessing Started")
            df = DataPreprocessing.preprocess(df)
            logger.info(f"Preprocessing Completed. Shape={df.shape}")

            #################################
            # STEP 3 Feature Engineering
            #################################
            if self.categorical_columns:
                missing_cols = [
                    col for col in self.categorical_columns
                    if col not in df.columns
                ]
                if missing_cols:
                    logger.warning(f"Skipping Feature Encoding. Missing columns: {missing_cols}")
                else:
                    logger.info(f"Applying Feature Encoding on: {self.categorical_columns}")
                    df = FeatureEngineering.transform_features(
                        df=df,
                        categorical_cols=self.categorical_columns,
                        model_name=self.model_name
                    )

            #################################
            # STEP 4 Feature Alignment
            #################################
            logger.info("STEP 4 - Feature Alignment Started")
            feature_file = os.path.join(FEATURE_COLUMNS_DIR, f"{self.model_name}_feature_columns.pkl")
            feature_columns = joblib.load(feature_file)
            logger.info(f"Loaded Feature Columns File: {feature_file}")
            df = df.reindex(columns=feature_columns, fill_value=0)
            logger.info(f"Aligned Feature Shape: {df.shape}")

            #################################
            # STEP 5 Scaling
            #################################
            logger.info("STEP 5 - Feature Scaling Started")
            DataScaling.load_latest_scaler(model_name=self.model_name)
            df = DataScaling.transform(df)
            logger.info(f"Scaling Completed. Shape={df.shape}")

            #################################
            # STEP 6 Prediction
            #################################
            logger.info("STEP 6 - Prediction Started")
            prediction = PredictionPipeline.predict(data=df, model_name=self.model_name)
            logger.info(f"Prediction Result: {prediction}")
            logger.info("Inference Pipeline Completed Successfully")
            return prediction

        except Exception as e:
            logger.exception(f"Inference Pipeline Failed: {str(e)}")
            raise


if __name__ == "__main__":
    '''
    sample = {
        "tenure": 72,
        "monthlyCharges": 20,
        "totalCharges": 5000,
        "contractType": "Yearly",
        "techSupport": "Yes",
        "complaints": "No"
    }
    '''
    sample = {
      "medinc": 4.5,
      "houseage": 25,
      "averooms": 5.5,
      "avebedrms": 1.1,
      "population": 1200,
      "aveoccup": 3.0,
      "latitude": 34.05,
      "longitude": -118.25
    }

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    pipeline = InferencePipeline(
        model_name=config["model"]["name"],
        categorical_columns=config["feature_engineering"]["categorical_columns"]
    )

    result = pipeline.run(sample)
    print(result)
