# pipelines/training_pipeline.py

import os, yaml
import sys

import joblib

from src.ingestion.data_ingestion import DataIngestion
from src.preprocessing.data_preprocessing import DataPreprocessing
from src.validation.data_validation import DataValidation
from src.feature_engineering.feature_engineering import FeatureEngineering
from src.transformation.data_transformation import DataTransformation
from src.transformation.scaling import DataScaling
from src.training.model_factory import ModelFactory
from src.training.train_model import TrainModel
from src.evaluation.evaluate import EvaluateModel
from src.constants.global_constants import CONFIG_PATH, FEATURE_COLUMNS_DIR


class TrainingPipeline:

    def __init__(self, config_path):
        self.config_path = config_path
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def run(self):

        task_type = self.config["model"]["task_type"]
        target = self.config["data"]["target_column"].lower()
        drop_columns = [col.lower() for col in self.config["data"].get("drop_columns", [])]
        categorical_cols = [col.lower() for col in self.config.get("feature_engineering", {}).get("categorical_columns", [])]

        ###################################
        # STEP 1: DATA INGESTION
        ###################################
        ingestion = DataIngestion()
        df = ingestion.load_data(file_name=self.config["data"]["file_name"],
                                 folder=self.config["data"].get("folder", "raw"))

        ###################################
        # STEP 2: Preprocessing
        ###################################
        df = DataPreprocessing.preprocess(df)

        ###################################
        # STEP 3: VALIDATION
        ###################################
        validator = DataValidation()
        validation = validator.validate(df)

        if validation["status"] == "failed":
            raise ValueError(f"Validation Failed: {validation}")

        ###################################
        # STEP 3: FEATURE ENGINEERING
        ###################################
        fe = FeatureEngineering()
        df = fe.fit_features(df, categorical_cols=categorical_cols, model_name=self.config["model"]["name"])
        df = fe.encode_target(df, target_col=target, model_name=self.config["model"]["name"])

        ###################################
        # STEP 4: TRAIN TEST SPLIT
        ###################################
        splitter = DataTransformation()
        X_train, X_test, y_train, y_test = splitter.prepare_and_split(
                df=df, target_column=target, drop_columns=drop_columns, task_type=task_type,
                test_size=self.config["training"].get("test_size"),
                random_state=self.config["training"].get("random_state"))

        joblib.dump(
        X_train.columns.tolist(),
        os.path.join(FEATURE_COLUMNS_DIR, f"{self.config['model']['name']}_feature_columns.pkl"))

        ###################################
        # STEP 5: SCALING
        ###################################
        scaler_path = None
        if self.config["training"].get("scaler"):
            X_train, scaler_path = (DataScaling.fit_transform(X_train=X_train,
                    model_name=self.config["model"]["name"],
                    scaler_name=self.config["training"]["scaler"]))

            X_test = DataScaling.transform(X_test)

        ###################################
        # STEP 6: MODEL SELECTION
        ###################################
        model = ModelFactory.get_model(model_name=self.config["model"]["name"],
            task_type=self.config["model"]["task_type"],
            **self.config["model"].get("params", {}))

        ###################################
        # STEP 7: TRAIN MODEL
        ###################################
        trained_model, model_path = (TrainModel.train(model, X_train, y_train, model_name=self.config["model"]["name"]))

        ###################################
        # STEP 8: EVALUATION
        ###################################
        if task_type.lower() == "classification":
            metrics, metrics_path = (EvaluateModel.evaluate_classification(trained_model, X_test, y_test, self.config["model"]["name"]))
        else:
            metrics, metrics_path = (EvaluateModel.evaluate_regression(trained_model, X_test, y_test, self.config["model"]["name"]))

        ###################################
        # RETURN PIPELINE OUTPUTS
        ###################################
        return {
            "metrics": metrics,
            "metrics_path": metrics_path,
            "model_path": model_path,
            "scaler_path": scaler_path,
            "config_path": self.config_path
        }


if __name__ == "__main__":
    '''
    yaml_file = sys.argv[1]
    config_path = os.path.join(PROJECT_ROOT, "configs",yaml_file)

    pipeline = TrainingPipeline()
    result = pipeline.run()
    print(result)
    '''

    pipeline = TrainingPipeline(config_path=CONFIG_PATH)
    result = pipeline.run()
    print(result)
