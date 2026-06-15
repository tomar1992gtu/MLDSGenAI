import yaml
import pandas as pd

from pydantic import Field, create_model
from src.constants.global_constants import CONFIG_PATH, DATA_DIR
from src.logging.logger import logger


def load_config():
    logger.info(f"Loading configuration from {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    logger.info("Configuration loaded successfully")
    return config


def build_prediction_schema():
    logger.info("Building dynamic prediction schema")
    config = load_config()
    target = config["data"]["target_column"].lower()
    drop_columns = [col.lower() for col in config["data"].get("drop_columns", [])]
    categorical_columns = [col.lower() for col in config.get("feature_engineering", {}).get("categorical_columns", [])]
    ignore_columns = set(drop_columns + [target])

    data_file = config["data"]["file_name"]

    if data_file.endswith(".csv"):
        df = pd.read_csv(f"{DATA_DIR}/raw/{data_file}", nrows=5)
    else:
        df = pd.read_excel(f"{DATA_DIR}/raw/{data_file}", nrows=5)

    df.columns = df.columns.str.strip().str.lower()

    schema_fields = {}
    logger.info(f"Schema source file loaded: {data_file}")
    logger.info(f"Columns detected: {df.columns.tolist()}")

    for col in df.columns:
        if col in ignore_columns:
            continue
        alias_name = col

        # categorical field
        if col in categorical_columns:
            schema_fields[col] = (str,
                Field(
                    ...,
                    alias=alias_name,
                    description=f"{col}"
                )
            )

        # numeric field
        else:
            dtype = df[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                schema_fields[col] = (int,
                    Field(
                        ...,
                        alias=alias_name,
                        description=f"{col}"
                    )
                )

            else:
                schema_fields[col] = (float,
                    Field(
                        ...,
                        alias=alias_name,
                        description=f"{col}"
                    )
                )

    DynamicSchema = create_model(
        "PredictionRequest",
        **schema_fields
    )

    DynamicSchema.model_config = {
        "populate_by_name": True
    }

    logger.info(f"Prediction schema created with {len(schema_fields)} fields")
    return DynamicSchema
