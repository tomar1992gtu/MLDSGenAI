# pipelines/training_pipeline.py

import os, yaml
import sys

import joblib

from src.ingestion.data_ingestion import DataIngestion
from src.logging.logger import logger
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

    logger.info(f"=====>>> Starting Training Pipeline <<======")

    def __init__(self, config_path):
        self.config_path = config_path
        logger.info(f"Loading configuration: {config_path}")

        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        logger.info("Configuration loaded successfully")

    def run(self):
        try:
            task_type = self.config["model"]["task_type"]
            model_name = self.config["model"]["name"]
            target = self.config["data"]["target_column"].lower()
            drop_columns = [col.lower() for col in self.config["data"].get("drop_columns", [])]
            categorical_cols = [col.lower() for col in self.config.get("feature_engineering", {}).get("categorical_columns", [])]

            logger.info(f"Model: {model_name} | Task: {task_type}")
            ###################################
            # STEP 1: DATA INGESTION
            ###################################
            ingestion = DataIngestion()
            logger.info("STEP 1 - Data Ingestion Started")
            df = ingestion.load_data(file_name=self.config["data"]["file_name"],
                                     folder=self.config["data"].get("folder", "raw"))
            logger.info(f"Dataset Loaded Successfully. Shape={df.shape}")

            ###################################
            # STEP 2: Preprocessing
            ###################################
            logger.info("STEP 2 - Data Preprocessing Started")
            df = DataPreprocessing.preprocess(df)
            logger.info(f"Preprocessing Completed. Shape={df.shape}")

            ###################################
            # STEP 3: VALIDATION
            ###################################
            validator = DataValidation()
            logger.info("STEP 3 - Data Validation Started")
            validation = validator.validate(df)

            if validation["status"] == "failed":
                raise ValueError(f"Validation Failed: {validation}")
            logger.info(f"Validation Status: {validation['status']}")

            ###################################
            # STEP 4: FEATURE ENGINEERING
            ###################################
            fe = FeatureEngineering()
            logger.info("STEP 4 - Feature Engineering Started")
            df = fe.fit_features(df, categorical_cols=categorical_cols, model_name=model_name)
            if task_type.lower() == "classification":
                df = fe.encode_target(df, target_col=target, model_name=model_name)
            logger.info(f"Feature Engineering Completed. Shape={df.shape}")

            ###################################
            # STEP 5: TRAIN TEST SPLIT
            ###################################
            splitter = DataTransformation()
            logger.info("STEP 5 - Train Test Split Started")
            X_train, X_test, y_train, y_test = splitter.prepare_and_split(
                    df=df, target_column=target, drop_columns=drop_columns, task_type=task_type,
                    test_size=self.config["training"].get("test_size"),
                    random_state=self.config["training"].get("random_state"))
            logger.info(f"X_train={X_train.shape}, X_test={X_test.shape}")
            os.makedirs(FEATURE_COLUMNS_DIR, exist_ok=True)
            feature_file = os.path.join(FEATURE_COLUMNS_DIR, f"{model_name}_feature_columns.pkl")
            joblib.dump(X_train.columns.tolist(), feature_file)
            logger.info(f"Saving Feature Columns: {feature_file}")

            ###################################
            # STEP 6: SCALING
            ###################################
            scaler_path = None
            logger.info("STEP 6 - Feature Scaling Started")
            if self.config["training"].get("scaler"):
                X_train, scaler_path = (DataScaling.fit_transform(X_train=X_train,
                        model_name=model_name,
                        scaler_name=self.config["training"]["scaler"]))

                X_test = DataScaling.transform(X_test)
                logger.info(f"Scaler Saved: {scaler_path}")

            ###################################
            # STEP 7: MODEL SELECTION
            ###################################
            logger.info(f"Creating Model: {model_name}")
            logger.info("STEP 7 - Model Training Started")
            model = ModelFactory.get_model(model_name=model_name,
                task_type=self.config["model"]["task_type"],
                **self.config["model"].get("params", {}))

            ###################################
            # STEP 8: TRAIN MODEL
            ###################################
            logger.info("STEP 8 - Model Evaluation Started")
            trained_model, model_path = (TrainModel.train(model, X_train, y_train, model_name=model_name))
            logger.info(f"Model Saved: {model_path}")

            ###################################
            # STEP 9: EVALUATION
            ###################################
            if task_type.lower() == "classification":
                metrics, metrics_path = (EvaluateModel.evaluate_classification(trained_model, X_test, y_test, model_name))
            else:
                metrics, metrics_path = (EvaluateModel.evaluate_regression(trained_model, X_test, y_test, model_name))
            logger.info(f"STEP 9 - Metrics: {metrics}")

            ###################################
            # RETURN PIPELINE OUTPUTS
            ###################################
            logger.info("Training Pipeline Completed Successfully")
            return {
                "metrics": metrics,
                "metrics_path": metrics_path,
                "model_path": model_path,
                "scaler_path": scaler_path,
                "config_path": self.config_path
            }

        except Exception as e:
            logger.exception(f"Training Pipeline Failed: {str(e)}")
            raise


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
