import pandas as pd

from src.logging.logger import logger


class DataPreprocessing:

    @classmethod
    def preprocess(cls, df: pd.DataFrame, remove_duplicates=True, strip_columns=True, fill_missing=True):
        try:
            logger.info("Data Preprocessing Started")
            df = df.copy()

            logger.info(f"Initial Shape: {df.shape}")
            # 1. Strip Column Names
            if strip_columns:
                logger.info("Standardizing column names")
                df.columns = (df.columns.str.strip().str.lower())
                #df.columns = (df.columns.str.strip())

            # 2. Remove Duplicate Rows
            if remove_duplicates:
                before_rows = len(df)
                df = df.drop_duplicates()
                removed_rows = before_rows - len(df)
                logger.info(f"Duplicate rows removed: {removed_rows}")

            # 3. Handle Missing Values
            if fill_missing:
                logger.info("Handling missing values")
                numeric_cols = df.select_dtypes(include=["number"]).columns
                categorical_cols = df.select_dtypes(exclude=["number"]).columns

                # numeric → median
                for col in numeric_cols:
                    missing_count = df[col].isna().sum()
                    if missing_count > 0:
                        logger.info(f"Filled {missing_count} missing values in numeric column: {col}")
                        df[col] = df[col].fillna(df[col].median())

                # categorical → mode
                for col in categorical_cols:
                    missing_count = df[col].isna().sum()
                    if missing_count > 0:
                        logger.info(f"Filled {missing_count} missing values in categorical column: {col}")
                    if not df[col].mode().empty:
                        df[col] = df[col].fillna(df[col].mode()[0])

            # 4. Strip String Values
            object_cols = df.select_dtypes(include="object").columns
            for col in object_cols:
                df[col] = (df[col].astype(str).str.strip())
            logger.info(f"Final Shape: {df.shape}")
            logger.info("Data Preprocessing Completed Successfully")

            return df

        except Exception:
            logger.exception("Data Preprocessing Failed")
            raise
