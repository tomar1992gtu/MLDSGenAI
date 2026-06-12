import json
import os

from datetime import datetime
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score)

from src.constants.global_constants import ARTIFACTS_METRICS_DIR


class EvaluateModel:

    @classmethod
    def save_metrics(cls, metrics: dict, model_name: str):
        os.makedirs(ARTIFACTS_METRICS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"{ARTIFACTS_METRICS_DIR}/{model_name}_metrics_{timestamp}.json"

        with open(file_path, "w") as f:
            json.dump(metrics, f, indent=4)

        return file_path

    # -------------------------
    # Classification
    # -------------------------
    @classmethod
    def evaluate_classification(cls, model, X_test, y_test, model_name="model"):

        predictions = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, average="weighted"),
            "recall": recall_score(y_test, predictions, average="weighted"),
            "f1_score": f1_score(y_test, predictions, average="weighted"),
        }

        file_path = cls.save_metrics(metrics, model_name)

        return metrics, file_path

    # -------------------------
    # Regression
    # -------------------------
    @classmethod
    def evaluate_regression(cls, model, X_test, y_test, model_name="model"):

        predictions = model.predict(X_test)

        metrics = {
            "rmse": mean_squared_error(y_test, predictions, squared=False),
            "r2_score": r2_score(y_test, predictions)
        }

        file_path = cls.save_metrics(metrics, model_name)

        return metrics, file_path


'''
Usages :
--------
metrics, file_path = EvaluateModel.evaluate_regression(model, X_test, y_test, model_name="LinearRegression")
print(metrics)
print(file_path)

metrics, file_path = EvaluateModel.evaluate_classification(model, X_test, y_test, model_name="RandomForestClassifier")
print(metrics)
print(file_path)
'''
