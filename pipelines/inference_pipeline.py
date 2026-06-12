import os

import joblib
import pandas as pd
import yaml

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

    def run(self, input_data):

        #################################
        # STEP 1 Convert input
        #################################
        if not isinstance(input_data, pd.DataFrame):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()

        #################################
        # STEP 2 Preprocessing
        #################################
        df = DataPreprocessing.preprocess(df)

        #################################
        # STEP 3 Feature Engineering
        #################################
        fe = FeatureEngineering()
        if self.categorical_columns:
            df = fe.transform_features(df=df, categorical_cols=self.categorical_columns, model_name=self.model_name)

        feature_columns = joblib.load(os.path.join(FEATURE_COLUMNS_DIR, f"{self.model_name}_feature_columns.pkl"))
        df = df.reindex(columns=feature_columns, fill_value=0)

        #################################
        # STEP 4 Scaling
        #################################
        DataScaling.load_latest_scaler(model_name=self.model_name)
        df = DataScaling.transform(df)

        #################################
        # STEP 5 Prediction
        #################################
        prediction = PredictionPipeline.predict(data=df, model_name=self.model_name)
        return prediction


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
