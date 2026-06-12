# 📊 Data Science Project – End-to-End ML Pipeline

This repository contains an **end-to-end data science pipeline** designed to handle multiple machine learning use cases.
Currently, it supports:

* 🏠 **California Housing Price Prediction**

---

## 📁 Project Structure

```
MLDSGenAI/
│
├── data/                           # Stores all datasets used throughout the project
│   ├── raw/                        # Original untouched data collected from sources
│   ├── external/                   # Third-party or externally obtained datasets
│   └── processed/                  # Cleaned and transformed datasets ready for modeling
│
├── configs/                        # Configuration files for different projects/models
│   ├── base.yaml                   # Common configuration shared across projects
│   ├── housing.yaml                # Housing project specific configuration
│   ├── titanic.yaml                # Titanic project specific configuration
│
├── src/                            # Core source code of the framework
│   ├── constants/                  # Define Constants Values
│   ├── common_lib/                 # Define Reusable methods
│   ├── ingestion/                  # Data collection and loading logic
│   ├── preprocessing/              # Data Preprocessing
│   ├── validation/                 # Data validation and schema checking
│   ├── transformation/             # Data cleaning and preprocessing logic
│   ├── feature_engineering/        # Feature creation and selection logic
│   ├── training/                   # Model training related components
│   ├── evaluation/                 # Model evaluation and performance analysis
│   ├── inference/                  # Prediction and inference related logic
│   ├── monitoring/                 # Monitoring model performance and drift
│   ├── registry/                   # Model registration and version management
│   ├── logging/                    # Logging utilities and configurations
│   ├── exceptions/                 # Custom exception handling classes
│   └── utils/                      # Reusable helper functions and utilities
│
├── pipelines/                      # End-to-end workflow orchestration
│   ├── training_pipeline.py        # Complete training workflow pipeline
│   └── inference_pipeline.py       # Complete inference workflow pipeline
│
├── app/                            # Application layer for deployment/services
│   ├── api/                        # API routes and endpoint definitions
│   ├── services/                   # Business logic and service layer
│   └── main.py                     # Application entry point
│
├── experiments/                    # Experimental work, research, and trials
│
├── artifacts/                      # Generated outputs and project artifacts
│   ├── metrics/                    # Saved evaluation metrics
│   ├── reports/                    # Generated reports and summaries
│
├── models/                         # Stores trained model related files
│   ├── trained_models/             # Serialized trained models
│   ├── encoders/                   # Saved encoders and preprocessing objects
│
├── tests/                          # Unit tests and integration tests
│
├── requirements.txt                # Project dependencies
├── README.md                       # Project documentation
└── main.py                         # Main execution entry point

```

---

## 🔧 Features

✔ Modular & scalable architecture
✔ Config-driven pipeline (YAML-based)
✔ Supports regression & classification
✔ Automated data validation
✔ Feature scaling & encoding
✔ Model factory pattern
✔ Evaluation metrics & comparison
✔ Model & data drift monitoring
✔ API-ready for deployment

---

## 🏠 California Housing Project

**Goal:** Predict median house prices based on demographic and geographic features.

**Key Techniques:**

* Linear & non-linear regression models
* Feature scaling and multicollinearity handling (VIF)
* Model evaluation using RMSE, MAE, R²

---

## ⚙️ Configuration Management

All experiments are **config-driven** using YAML files:

* `params.yaml` → model & training parameters
* `schema.yaml` → TBC === data schema & validation rules
* `default.yaml` → TBC === shared global settings

This allows easy experimentation without changing code.

---

## 🚀 Running the Training Pipeline

```bash
python pipelines/training_pipeline.py
```

The pipeline will:

1. Ingest raw data
2. Validate schema
3. Transform features
4. Train model
5. Evaluate performance
6. Save artifacts

---

## 📈 Monitoring

* **Data Drift:** Detects changes in incoming data distribution
* **Model Drift:** Monitors degradation in model performance

Located in:

```
monitoring/
├── data_drift.py
└── model_drift.py
```

---

## 🌐 API Deployment

A simple inference API is available:

```
app/api.py
```

This can be extended using **FastAPI** or **Flask** for real-time predictions.

---

## 📦 Artifacts

All trained models, encoders, and scalers are stored in:

```
artifacts/
├── housing/
└── titanic/
```

Versioning ensures reproducibility and traceability.

---

## 🛠 Tech Stack

* Python
* Pandas, NumPy
* Scikit-Learn
* YAML
* Matplotlib / Seaborn
* FastAPI / Flask (optional)

---

## 📌 Future Enhancements

* MLflow integration
* CI/CD pipelines
* Docker support
* Cloud deployment (AWS / GCP / Azure)
* Automated hyperparameter tuning

---

## 👤 Author

**Jitendra TOMAR**
Data Scientist | Machine Learning Engineer

---

## 📄 License

This project is licensed under the **MIT License**.
