import traceback

from fastapi import APIRouter
import yaml

from app.schemas.schema_builder import build_prediction_schema
from pipelines.inference_pipeline import InferencePipeline
from src.constants.global_constants import CONFIG_PATH
from src.logging.logger import logger


with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

PredictionRequest = build_prediction_schema()
router = APIRouter()


@router.post("/predict")
def predict(request: PredictionRequest):

    try:
        logger.info("Prediction Request Received")
        input_data = request.model_dump(by_alias=False)
        logger.info(f"Input Data: {input_data}")

        pipeline = InferencePipeline(
            model_name=config["model"]["name"],
            categorical_columns=config.get("feature_engineering", {}).get("categorical_columns", []))

        prediction_result = pipeline.run(input_data=input_data)

        response = {
            "status": "success",
            "model": config["model"]["name"],
            "prediction": prediction_result["prediction"][0]
        }

        if "probability" in prediction_result:
            response["probability"] = (prediction_result["probability"][0])

        logger.info(f"Prediction Response: {response}")
        return response

    except Exception as e:
        logger.exception(f"Prediction Failed: {str(e)}")
        return {
            "status": "failed",
            "error": str(e),
            "error_type": type(e).__name__
        }
