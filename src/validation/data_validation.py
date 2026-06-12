import pandas as pd

class DataValidation:

    def validate(self, df: pd.DataFrame, raise_error: bool = False):
        errors = []

        # 1. Missing values check
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0]

        if not missing_cols.empty:
            errors.append({
                "issue": "missing_values",
                "details": missing_cols.to_dict()
            })

        # 2. Duplicate check
        duplicate_count = df.duplicated().sum()

        if duplicate_count > 0:
            errors.append({
                "issue": "duplicate_rows",
                "count": int(duplicate_count)
            })

        # 3. Empty dataframe check
        if df.empty:
            errors.append({
                "issue": "empty_dataframe"
            })

        # 4. Final decision
        if errors and raise_error:
            raise ValueError(f"Data validation failed: {errors}")

        return {
            "status": "failed" if errors else "passed",
            "errors": errors
        }
