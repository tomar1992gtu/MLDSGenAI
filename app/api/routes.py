import traceback

from fastapi import APIRouter
import yaml

from app.schemas.schema_builder import build_prediction_schema
from pipelines.inference_pipeline import InferencePipeline
from src.constants.global_constants import CONFIG_PATH


# ==========================================
# Load Config
# ==========================================
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


# ==========================================
# Dynamic Request Schema
# ==========================================
PredictionRequest = build_prediction_schema()

# ==========================================
# Router
# ==========================================
router = APIRouter()

# ==========================================
# Prediction Endpoint
# ==========================================
@router.post("/predict")
def predict(request: PredictionRequest):

    try:
        input_data = request.model_dump(
            by_alias=False
        )

        pipeline = InferencePipeline(
            model_name=config["model"]["name"],
            categorical_columns=config.get("feature_engineering", {}).get("categorical_columns", []))

        prediction = pipeline.run(input_data=input_data)

        return {
            "status": "success",
            "model": config["model"]["name"],
            "prediction": prediction["prediction"]
        }

    except Exception as e:
        traceback.print_exc()

        return {
            "status": "failed",
            "error": str(e),
            "error_type": type(e).__name__
    }
