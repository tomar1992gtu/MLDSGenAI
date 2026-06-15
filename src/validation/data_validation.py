import pandas as pd

from src.logging.logger import logger


class DataValidation:

    @classmethod
    def validate(cls, df: pd.DataFrame, raise_error: bool = False):
        try:
            logger.info("Data Validation Started")
            errors = []

            # 1. Missing values check
            missing = df.isnull().sum()
            missing_cols = missing[missing > 0]

            if not missing_cols.empty:
                logger.warning(f"Missing values found in columns: {missing_cols.to_dict()}")
                errors.append({
                    "issue": "missing_values",
                    "details": missing_cols.to_dict()
                })
            else:
                logger.info("No missing values found")

            # 2. Duplicate check
            duplicate_count = df.duplicated().sum()

            if duplicate_count > 0:
                logger.warning(f"Duplicate rows found: {duplicate_count}")
                errors.append({
                    "issue": "duplicate_rows",
                    "count": int(duplicate_count)
                })
            else:
                logger.info("No duplicate rows found")

            # 3. Empty dataframe check
            if df.empty:
                logger.error("DataFrame is empty")
                errors.append({
                    "issue": "empty_dataframe"
                })
            else:
                logger.info(f"DataFrame contains {len(df)} rows and {len(df.columns)} columns")

            # 4. Final decision
            status = "failed" if errors else "passed"
            logger.info(f"Validation Status: {status}")

            if errors and raise_error:
                logger.error(f"Validation Failed: {errors}")
                raise ValueError(f"Data validation failed: {errors}")

            logger.info("Data Validation Completed")
            return {
                "status": status,
                "errors": errors
            }

        except Exception:
            logger.exception("Data Validation Failed")
            raise
