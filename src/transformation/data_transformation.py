from sklearn.model_selection import train_test_split


class DataTransformation:

    def prepare_and_split(self, df, target_column, task_type, drop_columns=None, test_size=0.2, random_state=42):

        if drop_columns is None:
            drop_columns = []

        columns_to_remove = drop_columns + [target_column]

        X = df.drop(columns=columns_to_remove, errors="ignore")
        y = df[target_column]

        stratify_value = (y if task_type.lower() == "classification" else None)

        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_value
        )
