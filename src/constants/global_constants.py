import os

# ==========================
# PROJECT ROOT
# ==========================
PROJECT_ROOT = os.path.dirname(__file__).split('src')[0]

# ==========================
# DATA PATHS
# ==========================
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# ==========================
# MODEL PATHS
# ==========================
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
ENCODER_DIR = os.path.join(MODELS_DIR, "encoders")
TRAINED_MODELS_DIR = os.path.join(MODELS_DIR, "trained_models")

# ==========================
# ARTIFACTS
# ==========================
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
ARTIFACTS_ENCODER_DIR = os.path.join(ARTIFACTS_DIR, "encoders")
ARTIFACTS_METRICS_DIR = os.path.join(ARTIFACTS_DIR, "metrics")
FEATURE_COLUMNS_DIR = os.path.join(ARTIFACTS_DIR, "feature_columns")

# ==========================
# CONFIG
# ==========================
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")
CONFIG_NAME = os.getenv("CONFIG_NAME")
#CONFIG_NAME = "customer_churn_params.yaml"
#CONFIG_NAME = "housing_params.yaml"
CONFIG_PATH = os.path.join(CONFIG_DIR, CONFIG_NAME)
