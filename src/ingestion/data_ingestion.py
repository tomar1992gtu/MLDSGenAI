import os
import pandas as pd
from src.constants.global_constants import DATA_DIR, RAW_DATA_DIR


class DataIngestion:

    def load_data(self, file_name, folder=RAW_DATA_DIR):
        # Full file path
        file_path = os.path.join(DATA_DIR, folder, file_name)

        # File extension
        extension = os.path.splitext(file_name)[1]

        # Dynamic reader
        if extension == ".csv":
            df = pd.read_csv(file_path)
        elif extension in [".xls", ".xlsx"]:
            df = pd.read_excel(file_path)
        elif extension == ".json":
            df = pd.read_json(file_path)
        elif extension == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")

        return df



