# 📊 MLDSGenAI – End-to-End Machine Learning Framework

A configurable and reusable Machine Learning framework for building, training, evaluating, and deploying both **Regression** and **Classification** models.

The framework follows a modular architecture and supports:

* Data Ingestion
* Data Validation
* Data Preprocessing
* Feature Engineering
* Feature Scaling
* Model Training
* Model Evaluation
* Model Persistence
* Prediction APIs using FastAPI
* Logging & Monitoring of Pipeline Execution

---

# 🚀 Supported Use Cases

## Classification

Examples:

* Customer Churn Prediction
* Loan Approval Prediction
* Employee Attrition Prediction
* Titanic Survival Prediction

## Regression

Examples:

* California Housing Price Prediction
* House Price Prediction
* Sales Forecasting
* Demand Prediction

---

# 📂 Project Structure

```text
MLDSGenAI/
│
├── app/
│   ├── api/
│   │   └── routes.py
│   ├── schemas/
│   │   └── schema_builder.py
│   └── main.py
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── pipelines/
│   ├── training_pipeline.py
│   └── inference_pipeline.py
│
├── src/
│   ├── constants/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── validation/
│   ├── feature_engineering/
│   ├── transformation/
│   ├── training/
│   ├── evaluation/
│   ├── prediction/
│   └── logging/
│
├── artifacts/
│   ├── trained_models/
│   ├── encoders/
│   ├── metrics/
│   ├── logs/
│   └── feature_columns/
│
├── requirements.txt
├── README.md
└── main.py
```

---

# ⚙️ Framework Workflow

```text
Raw Dataset
      │
      ▼
Data Ingestion
      │
      ▼
Data Preprocessing
      │
      ▼
Data Validation
      │
      ▼
Feature Engineering
      │
      ▼
Train/Test Split
      │
      ▼
Feature Scaling
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Artifact Storage
      │
      ▼
Prediction API
```

---

# 🔧 Features

### Data Ingestion

Supports loading:

* CSV
* Excel (.xlsx)
* JSON
* Parquet

---

### Data Preprocessing

Automatically handles:

* Column name standardization
* Duplicate removal
* Missing value treatment
* String cleaning

---

### Data Validation

Validates:

* Missing values
* Duplicate rows
* Empty datasets

---

### Feature Engineering

Supports:

* One-Hot Encoding
* Label Encoding
* Feature persistence for inference

---

### Feature Scaling

Available scalers:

* StandardScaler
* MinMaxScaler
* RobustScaler

Saved automatically and reused during inference.

---

### Model Factory

Supports multiple models.

#### Regression Models

* LinearRegression
* Ridge
* Lasso
* ElasticNet
* DecisionTreeRegressor
* RandomForestRegressor
* GradientBoostingRegressor
* KNeighborsRegressor
* SVR
* MLPRegressor
* XGBRegressor

#### Classification Models

* LogisticRegression
* DecisionTreeClassifier
* RandomForestClassifier
* GradientBoostingClassifier
* AdaBoostClassifier
* KNeighborsClassifier
* SVC
* MLPClassifier
* GaussianNB
* XGBClassifier

---

### Model Evaluation

#### Classification Metrics

* Accuracy
* Precision
* Recall
* F1 Score

#### Regression Metrics

* RMSE
* R² Score

Metrics are automatically saved as JSON artifacts.

---

### Logging

Every pipeline step is logged.

Example:

```text
2026-06-15 12:45:10 - INFO - Data Ingestion Started
2026-06-15 12:45:12 - INFO - Dataset Loaded Successfully
2026-06-15 12:45:14 - INFO - Model Training Started
2026-06-15 12:45:18 - INFO - Model Saved Successfully
```

Logs are stored in:

```text
artifacts/logs/
```

---

# 📄 Configuration Driven Architecture

All pipeline behavior is controlled through YAML configuration.

Example:

```yaml
data:
  file_name: customer_churn.xlsx
  target_column: churn

feature_engineering:
  categorical_columns:
    - contracttype
    - techsupport

training:
  test_size: 0.2
  random_state: 42
  scaler: standard

model:
  name: RandomForestClassifier
  task_type: Classification
```

No code changes are required to switch models or datasets.

---

# 🚀 Run Training Pipeline

```bash
python main.py --mode train --config configs/config.yaml
```

---

# 🔮 Run Inference Pipeline

```bash
python main.py --mode predict --config configs/config.yaml
```

---

# 🌐 Start FastAPI Server

```bash
uvicorn app.main:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

# 📦 Generated Artifacts

```text
artifacts/
│
├── trained_models/
├── encoders/
├── metrics/
├── logs/
└── feature_columns/
```

Each training run automatically stores:

* Model
* Encoders
* Scalers
* Metrics
* Feature Columns
* Logs

---

# 🛠 Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* FastAPI
* Pydantic
* Joblib
* YAML

---

# 🔮 Future Enhancements

* MLflow Integration
* Docker Support
* CI/CD Pipelines
* Hyperparameter Tuning
* Feature Store
* Model Registry
* Cloud Deployment (AWS / Azure / GCP)

---

# 👨‍💻 Author

**Jitendra Tomar**

Machine Learning | Data Science | Generative AI

GitHub:
https://github.com/gtu12tomar

---

# 📜 License

This project is licensed under the MIT License.
