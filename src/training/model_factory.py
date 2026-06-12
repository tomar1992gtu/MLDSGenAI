# -------------------------
# Linear Models
# -------------------------
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression)

# -------------------------
# Tree-Based Models
# -------------------------
from sklearn.tree import (DecisionTreeRegressor, DecisionTreeClassifier)
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostClassifier)

# -------------------------
# Neighbors
# -------------------------
from sklearn.neighbors import (KNeighborsRegressor, KNeighborsClassifier)

# -------------------------
# Support Vector Machines
# -------------------------
from sklearn.svm import (SVR, SVC)

# -------------------------
# Neural Networks
# -------------------------
from sklearn.neural_network import (MLPRegressor, MLPClassifier)

# -------------------------
# Naive Bayes
# -------------------------
from sklearn.naive_bayes import GaussianNB

# -------------------------
# XGBoost
# -------------------------
from xgboost import (XGBRegressor, XGBClassifier)

'''
Purpose : Instead of manually creating models everywhere, create them from one centralized place.
'''

class ModelFactory:

    # =====================================
    # REGRESSION MODELS
    # =====================================
    REGRESSION_MODELS = {
        "LinearRegression": LinearRegression,
        "Ridge": Ridge,
        "Lasso": Lasso,
        "ElasticNet": ElasticNet,
        "DecisionTreeRegressor": DecisionTreeRegressor,
        "RandomForestRegressor": RandomForestRegressor,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "KNeighborsRegressor": KNeighborsRegressor,
        "SVR": SVR,
        "MLPRegressor": MLPRegressor,
        "XGBRegressor": XGBRegressor
    }

    # =====================================
    # CLASSIFICATION MODELS
    # =====================================
    CLASSIFICATION_MODELS = {
        "LogisticRegression": LogisticRegression,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "AdaBoostClassifier": AdaBoostClassifier,
        "KNeighborsClassifier": KNeighborsClassifier,
        "SVC": SVC,
        "MLPClassifier": MLPClassifier,
        "GaussianNB": GaussianNB,
        "XGBClassifier": XGBClassifier
    }

    # =====================================
    # GET MODEL
    # =====================================
    # **kwargs --> Allows dynamic hyperparameters: n_estimators=200, max_depth=10
    @classmethod
    def get_model(cls, model_name: str, task_type: str = "classification", **kwargs):

        # Select model group
        if task_type.lower() == "classification":
            models = cls.CLASSIFICATION_MODELS
        elif task_type.lower() == "regression":
            models = cls.REGRESSION_MODELS
        else:
            raise ValueError("task_type must be either 'classification' or 'regression'")

        # Validate model name
        if model_name not in models:
            raise ValueError(f"Unsupported model: {model_name}\n"
                f"Available models: {list(models.keys())}"
            )

        # Create model instance
        return models[model_name](**kwargs)

    # =====================================
    # LIST AVAILABLE MODELS
    # =====================================
    @classmethod
    def list_models(cls):

        return {
            "classification": list(cls.CLASSIFICATION_MODELS.keys()),
            "regression": list(cls.REGRESSION_MODELS.keys())
        }

