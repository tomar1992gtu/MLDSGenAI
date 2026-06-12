import yaml
import pandas as pd

from pydantic import BaseModel, Field, create_model
from src.constants.global_constants import CONFIG_PATH, DATA_DIR


def load_config():

    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)


def build_prediction_schema():
    config = load_config()
    target = config["data"]["target_column"]
    drop_columns = config["data"].get("drop_columns", [])
    categorical_columns = config.get("feature_engineering", {}).get("categorical_columns", [])
    ignore_columns = set(drop_columns + [target])

    schema_fields = {}
    data_file = config["data"]["file_name"]

    if data_file.endswith(".csv"):
        df = pd.read_csv(f"{DATA_DIR}/raw/{data_file}", nrows=5)
    else:
        df = pd.read_excel(f"{DATA_DIR}/raw/{data_file}", nrows=5)

    schema_fields = {}

    for col in df.columns:
        if col in ignore_columns:
            continue
        alias_name = col.lower()

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

    return DynamicSchema
