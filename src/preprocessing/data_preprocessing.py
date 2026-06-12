import pandas as pd


class DataPreprocessing:

    @classmethod
    def preprocess(cls, df: pd.DataFrame, remove_duplicates=True, strip_columns=True, fill_missing=True):

        df = df.copy()

        # 1. Strip Column Names
        if strip_columns:
            df.columns = (df.columns.str.strip().str.lower())
            #df.columns = (df.columns.str.strip())

        # 2. Remove Duplicate Rows
        if remove_duplicates:
            df = df.drop_duplicates()

        # 3. Handle Missing Values
        if fill_missing:
            numeric_cols = df.select_dtypes(include=["number"]).columns
            categorical_cols = df.select_dtypes(exclude=["number"]).columns

            # numeric → median
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())

            # categorical → mode
            for col in categorical_cols:
                if not df[col].mode().empty:
                    df[col] = df[col].fillna(df[col].mode()[0])

        # 4. Strip String Values
        object_cols = df.select_dtypes(include="object").columns
        for col in object_cols:
            df[col] = (df[col].astype(str).str.strip())

        return df
