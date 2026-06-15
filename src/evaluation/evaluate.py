import json
import os

from datetime import datetime
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score)

from src.constants.global_constants import ARTIFACTS_METRICS_DIR
from src.logging.logger import logger


class EvaluateModel:

    @classmethod
    def save_metrics(cls, metrics: dict, model_name: str):
        try:
            os.makedirs(ARTIFACTS_METRICS_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"{ARTIFACTS_METRICS_DIR}/{model_name}_metrics_{timestamp}.json"

            with open(file_path, "w") as f:
                json.dump(metrics, f, indent=4)

            logger.info(f"Metrics Saved: {file_path}")
            return file_path

        except Exception:
            logger.exception("Failed To Save Metrics")
            raise

    # -------------------------
    # Classification
    # -------------------------
    @classmethod
    def evaluate_classification(cls, model, X_test, y_test, model_name="model"):
        try:
            logger.info(f"Classification Evaluation Started: {model_name}")
            logger.info(f"Test Shape: {X_test.shape}")
            predictions = model.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, average="weighted"),
                "recall": recall_score(y_test, predictions, average="weighted"),
                "f1_score": f1_score(y_test, predictions, average="weighted"),
            }

            logger.info(f"Classification Metrics: {metrics}")
            file_path = cls.save_metrics(metrics, model_name)
            logger.info("Classification Evaluation Completed")
            return metrics, file_path

        except Exception:
            logger.exception("Classification Evaluation Failed")
            raise


    # -------------------------
    # Regression
    # -------------------------
    @classmethod
    def evaluate_regression(cls, model, X_test, y_test, model_name="model"):
        try:
            logger.info(f"Regression Evaluation Started: {model_name}")
            logger.info(f"Test Shape: {X_test.shape}")
            predictions = model.predict(X_test)

            metrics = {
                "rmse": mean_squared_error(y_test, predictions, squared=False),
                "r2_score": r2_score(y_test, predictions)
            }
            logger.info(f"Regression Metrics: {metrics}")

            file_path = cls.save_metrics(metrics, model_name)
            logger.info("Regression Evaluation Completed")
            return metrics, file_path

        except Exception:
            logger.exception("Regression Evaluation Failed")
            raise
