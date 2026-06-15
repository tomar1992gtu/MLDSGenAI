import os
import pandas as pd
from src.logging.logger import logger
from src.constants.global_constants import DATA_DIR, RAW_DATA_DIR


class DataIngestion:

    def load_data(self, file_name, folder=RAW_DATA_DIR):
        try :
            # Full file path
            logger.info("DataIngestion Started")
            file_path = os.path.join(DATA_DIR, folder, file_name)
            logger.info(f"Loading file: {folder}/{file_name}")
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
                logger.error(f"Unsupported file format: {extension}")
                raise ValueError(f"Unsupported file format: {extension}")

            logger.info(f"Rows Loaded: {len(df)}")
            logger.info(f"Columns Loaded: {df.columns.tolist()}")
            logger.info("DataIngestion Finished")
            return df

        except Exception:
            logger.exception("Data Ingestion Failed")
            raise
