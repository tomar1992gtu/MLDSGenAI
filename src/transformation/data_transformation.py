from sklearn.model_selection import train_test_split

from src.logging.logger import logger


class DataTransformation:

    @classmethod
    def prepare_and_split(cls, df, target_column, task_type, drop_columns=None, test_size=0.2, random_state=42):
        try:
            logger.info("Data Transformation Started")
            if drop_columns is None:
                drop_columns = []

            logger.info(f"Target Column: {target_column}")
            logger.info(f"Drop Columns: {drop_columns}")

            columns_to_remove = drop_columns + [target_column]
            X = df.drop(columns=columns_to_remove, errors="ignore")
            y = df[target_column]

            logger.info(f"Feature Shape Before Split: {X.shape}")
            logger.info(f"Target Shape Before Split: {y.shape}")

            ###################################################
            # Classification -> Stratified Split
            ###################################################
            stratify_value = None
            if task_type.lower() == "classification":
                stratify_value = y
                logger.info("Using Stratified Train-Test Split")
                logger.info(f"Class Distribution:\n{y.value_counts().to_dict()}")

            ###################################################
            # Train Test Split
            ###################################################
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_value
            )
            logger.info(f"X_train Shape: {X_train.shape}")
            logger.info(f"X_test Shape: {X_test.shape}")
            logger.info(f"y_train Shape: {y_train.shape}")
            logger.info(f"y_test Shape: {y_test.shape}")
            logger.info("Data Transformation Completed Successfully")

            return (X_train, X_test, y_train, y_test)

        except Exception:
            logger.exception("Data Transformation Failed")
            raise
