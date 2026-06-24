# 💳 Enterprise Credit Card Fraud Detection System

An end-to-end, machine learning-driven analytics pipeline built to flag potentially fraudulent credit card transactions. By synthesizing historical transaction patterns, risk-based heuristics, and contextual device signals, this system provides a robust solution for minimizes financial risks while protecting consumer transaction workflows.

---

## 📌 Project Architecture

The pipeline processes raw data, engineers highly predictive domain features, trains multiple classifier variants, and provisions an interactive inference portal.

       [ Raw Dataset ]
              │
              ▼
     [ Data Preprocessing ]  ──► (Imputation, Scaling, Encoding)
              │
              ▼
   [ Feature Engineering ]   ──► (Behavioral, Velocity & Risk Ratios)
              │
              ▼
      [ Model Training ]     ──► (Logistic Regression, Decision Trees, Random Forest)
              │
              ▼
     [ Model Evaluation ]    ──► (Precision-Recall Optimization for Imbalanced Data)
              │
              ▼
  [ Hyperparameter Tuning ]  ──► (Grid/Random Search Cross-Validation)
              │
              ▼
    [ Serialized Model ]     ──► (Artifact preservation via Joblib)
              │
              ▼
   [ Streamlit Deployment ]  ──► (Real-Time Live Web Inference Interface)

---

## 📂 Repository Structure

The project layout follows production-grade modular design patterns, cleanly separating source code, data layers, generated artifacts, and visualization reports.

FRAUD_DETECTION_PROJECT/
│
├── data/
│   ├── processed/                     # Formatted, scaled, and split matrices
│   └── credit_card_fraud_10k.csv      # Raw baseline transaction dataset
│
├── logs/                              # Production & pipeline execution runtime logs
│   ├── evaluate_model.log
│   ├── fix_leakage.log
│   ├── train_model.log
│   └── tune_model.log
│
├── models/                            # Serialized model objects and pipeline states (.pkl / .json)
│   ├── best_model_fixed.pkl           # Leakage-corrected production model
│   ├── best_model.pkl                 # Initial baseline model
│   ├── fe_metadata.pkl                # Metadata mapping for feature engineering
│   ├── fe_preprocessor.pkl            # Feature engineering preprocessor state
│   ├── preprocessor.pkl               # Standard data cleaning/imputation pipeline state
│   ├── training_metadata_fixed.json   # Corrected pipeline training execution logs
│   ├── training_metadata.json         # Baseline training execution logs
│   ├── training_metadata.pkl          # Serialized training metadata metrics
│   └── tuned_model.pkl                # Optimally hyperparameter-tuned artifact
│
├── notebooks/
│   └── eda_analysis.ipynb             # Exploratory Data Analysis & statistical checking
│
├── reports/                           # Performance readouts and analytical metrics
│   ├── eda_figures/                   # Visual plots from exploratory data analysis
│   ├── figures/                       # Automatically generated evaluation plots
│   ├── leakage_audit/                 # Reports tracking data leakage metrics
│   ├── leakage_fix/                   # Documentation on validation leakage corrections
│   ├── classification_report.txt      # Text-based breakdown of precision/recall metrics
│   ├── evaluation_metrics.json        # Structured key-value performance indicators
│   ├── model_comparison.csv           # Performance matrix across different model types
│   ├── tuning_results.csv             # Raw parameter optimization iteration tables
│   └── tuning_summary.json            # Final overview of best hyperparameter spaces
│
├── src/                               # Modular processing codebase
│   ├── __pycache__/                   # Compiled Python runtime bytecode
│   ├── _init_.py                     # Package initialization marker
│   ├── audit_leakage.py               # Diagnostics script to detect validation data leakage
│   ├── debug_leak_check.py            # Troubleshooting module for target/feature cross-contamination
│   ├── evaluate_model.py              # Performance visualization and reporting
│   ├── feature_engineering.py         # Domain-driven feature engineering extraction
│   ├── fix_leakage.py                 # Routine to enforce absolute isolation between train/test splits
│   ├── predict.py                     # Batch/Independent sample inference script
│   ├── preprocessing.py               # Raw clean-up, scaling, and formatting logic
│   ├── saved_model.txt                # Tracking file for latest deployment-ready artifact versions
│   ├── train_model.py                 # Core model training execution pipeline
│   └── tune_model.py                  # Hyperparameter optimization sweeps
│
├── venv/                              # Isolated Python local virtual environment
├── app.py                             # Streamlit web UI production application entry-point
├── README.md                          # Systematic project manual
└── requirements.txt                   # Tracked third-party library dependencies

---

## 🔧 Core Stack & Technical Specifications

### Development Environment

* **Language:** Python 3.13
* **Deployment:** Streamlit Engine

### Core Engineering Libraries

* **Data Processing & Analytics:** Pandas, NumPy
* **Machine Learning Engine:** Scikit-Learn
* **Visual Diagnostics:** Matplotlib, Seaborn
* **Model Serialization:** Joblib

### Modeled Estimators

* Random Forest Classifier *(Selected Baseline)*
* Logistic Regression
* Decision Tree Classifier

---

## ⚙️ Engineered Features Domain Map

To maximize model sensitivity to fraudulent behavior, the following behavioral, temporal, and risk-stratified features were generated:

| Feature Name | Type | Technical Description |
| --- | --- | --- |
| `amount` | Numeric | Base value of the transaction |
| `transaction_hour` | Numeric | 24-hour timestamp indicator |
| `device_trust_score` | Numeric | Trust matrix metric calculated per device ID |
| `velocity_last_24h` | Numeric | Transaction frequency volume within a rolling 24-hour span |
| `cardholder_age` | Numeric | Account owner age profile |
| `amount_velocity_ratio` | Numeric | Compares financial volume spike against usage velocity |
| `foreign_transaction` | Boolean | Binary flag indicating out-of-country origin |
| `location_mismatch` | Boolean | Flags distance conflict between user profile and checkout terminal |
| `is_night_transaction` | Boolean | Flags high-risk off-hour windows (12 AM - 5 AM) |
| `is_high_amount` | Boolean | Threshold marker for statistical transaction outliers |
| `is_high_velocity` | Boolean | Flags excessive frequency anomalies |
| `is_low_device_trust` | Boolean | Flags hardware profiles categorized under insecure configurations |
| `is_foreign_mismatch` | Boolean | Evaluates compounding risk (foreign country + location mismatch) |
| `merchant_category_*` | Categorical | One-Hot encoded representations of target vendor sectors |

---

## 📊 Performance Benchmarks & Evaluation

### Validation Breakdown

* **Evaluation Split:** 80% Training / 20% Testing Validation sets
* **Total Evaluation Volume:** 2,000 Transactions *(1,970 Legitimate / 30 Fraudulent)*

The **Random Forest Classifier** outperformed all other structural variants, demonstrating elite capabilities in handling minority class variance.

### Global Metric Indicators

| Target Metric | Achieved Value |
| --- | --- |
| **Accuracy** | 99.75% |
| **Precision** | 99.74% |
| **Recall** | 99.75% |
| **F1-Score** | 99.74% |
| **ROC-AUC** | 99.99% |
| **Average Precision (AP)** | 99.06% |

## 🛡️ Rigorous Data Leakage Audit & Pipeline Isolation

In real-world fraud detection systems, data leakage is a critical risk that leads to over-optimistic validation metrics but failing production performance. During this project's lifecycle, a strict audit was executed using `src/audit_leakage.py` and resolved via `src/fix_leakage.py`.

### 🚨 Identified Vulnerabilities & Resolutions
* **Temporal Velocity Leakage:** Feature statistics like `velocity_last_24h` were initially calculated across the entire dataset globally before splitting. This leaked future transaction frequencies into past data points.
  * *Fix:* Rewrote the aggregation logic to compute rolling histories strictly partitioned within training folds.
* **Information Contamination via Feature Scaling:** Standardizing data matrices using global means and standard deviations leaks metadata from the testing validation set into the model's training phase.
  * *Fix:* Bound all preprocessing transformations within an explicit `sklearn.pipeline.Pipeline` instance, calling `.fit_transform()` exclusively on the training matrix and applying `.transform()` seamlessly to unseen transaction features.

As a result, the production artifacts (`best_model_fixed.pkl` and `training_metadata_fixed.json`) reflect an authentic, zero-leakage enterprise deployment profile.

### Detailed Class Segmentation


======================================================================
CLASS: LEGITIMATE TRANSACTIONS
----------------------------------------------------------------------
Precision: 0.997973 | Recall: 0.999492 | F1-Score: 0.998732

======================================================================
CLASS: FRAUDULENT TRANSACTIONS (Minority Class)
----------------------------------------------------------------------
Precision: 0.962963 | Recall: 0.866667 | F1-Score: 0.912281
======================================================================



> 💡 **Analytical Insight:** In highly imbalanced contexts like fraud detection, traditional accuracy is misleading. The model secures a **96.3% Precision** and an **86.7% Recall** on the minority fraud class, striking an optimal operational balance to catch fraud while maintaining low customer friction (false positives).

---

## 🚀 Environment Setup & Installation

### 1. Repository Acquisition


git clone https://github.com/your-username/fraud_detection_project.git
cd fraud_detection_project


### 2. Isolated Environment Construction

# Initialize Python Virtual Environment
python -m venv venv

# Activation: Windows Shell
venv\Scripts\activate

# Activation: macOS / Linux Terminal
source venv/bin/activate


### 3. Dependency Provisioning

pip install --upgrade pip
pip install -r requirements.txt


---

## 🏃 Execution Manual (Pipeline Orchestration)

Every phase of the pipeline can be individually executed or scheduled using modular scripts:

# 1. Cleanse and process raw input data matrices
python src/preprocessing.py

# 2. Extract domain-specific predictive heuristics 
python src/feature_engineering.py

# 3. Train core estimators and identify top configurations
python src/train_model.py

# 4. Generate visual metrics (Confusion Matrix, ROC/PR Curves)
python src/evaluate_model.py

# 5. Fine-tune hyperparameter spaces for top efficiency
python src/tune_model.py

---

## 🌐 Serving the Live Interactive Dashboard

The system features an intuitive user interface powered by Streamlit for instant ad-hoc assessments:

streamlit run app.py


Once initialized, navigate your local browser to the web terminal:

http://localhost:8501

---

## 🛡️ Risk Assessment Engine Logic

The model flag heuristic evaluates multi-tiered vectors. Risks compound and trigger fraud flags exponentially under conditions like:

* Extreme spikes in transaction values contrasted against baseline behavior.
* Mismatched tracking metrics (e.g., local home address vs. cross-border terminal pings).
* Highly suspect transactional velocity bounds from hardware profiles scoring poorly on security layers.

---

## 🔮 Strategic Enhancements Roadmap

* [ ] Transition baseline classifiers into gradient boosted frameworks (**XGBoost** and **LightGBM**).
* [ ] Implement asynchronous prediction streaming endpoints via a RESTful **FastAPI** structure.
* [ ] Containerize application infrastructure using **Docker** to facilitate seamless microservices management.
* [ ] Orchestrate automated continuous cloud deployment configurations (**AWS ECS / Azure App Services**).
* [ ] Integrate **SHAP (SHapley Additive exPlanations)** to establish highly interpretable and transparent ML decisions.
* [ ] Design automated data-drift monitors for trigger-driven programmatic pipeline retraining.

---

## 👩‍💻 Professional Profile

**Siva Priya A** *Bachelor of Engineering — Computer Science and Engineering*

**Core Technical Proficiencies & Focus Areas:**

* Artificial Intelligence & Predictive Modeling
* Enterprise Data Analytics Pipeline Engineering
* Full Stack Web Applications (MERN Ecosystem)
* Scalable Software Architecture

---

## 📜 Licensing

Distributed under the terms of the **MIT License**. See the project files for full terms and conditions regarding institutional use.