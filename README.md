# Loan-Default-Prediction-Model
Loan default prediction is a technique used in finance to assess the likelihood that a borrower will fail to repay a loan [1, 2, 3]. By analyzing borrower data, lenders can make informed decisions about loan approvals, interest rates, and loan terms [1, 4].

Here's a simplified overview of the process:

Data Collection: Lenders gather historical loan data including credit scores, employment history, and repayment history [1, 3].
Data Preprocessing: The data is cleaned and formatted for analysis [1].
Model Training: Machine learning algorithms are trained on the data to identify patterns and relationships between borrower characteristics and defaults [1, 4].
Model Evaluation: The model's performance is assessed to ensure its accuracy in predicting defaults [5].
Loan Application Scoring: New loan applications are scored based on the developed model to predict the likelihood of default [1].
Sources
https://www.analyticsvidhya.com/blog/2022/04/predicting-possible-loan-default-using-machine-learning/ Predicting Possible Loan Default Using Machine Learning
RIT Scholarly Works: https://repository.rit.edu/cgi/viewcontent.cgi?article=12544&context=theses Loan Default Prediction System by AA Ali Albastaki
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9552691/ Prediction and Analysis of Financial Default Loan Behavior ... by H Chen
Institute of Physics: https://iopscience.iop.org/article/10.1088/1757-899X/1022/1/012042/pdf Loan default prediction using decision trees and random ... by M Madaan
ScienceDirect: https://www.sciencedirect.com/science/article/abs/pii/S1544612323012394 Loan default predictability with explainable machine learning by H Li
ResearchGate: https://www.researchgate.net/publication/368807480_Loan_Default_Prediction_Model Loan Default Prediction Model



There are several Machine Learning models used to predict loan defaults, some of the most common include:

Logistic Regression: A statistical method that analyzes the relationship between borrower characteristics and loan default [5, 6] [NUMBER: https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning NUMBER: https://scholarworks.calstate.edu/downloads/8k71nk666].
Decision Trees: A tree-like model that classifies loan applications based on a series of borrower characteristics [3, 4] [NUMBER: https://www.scirp.org/journal/paperinformation.aspx?paperid=120102 NUMBER: [invalid URL removed]].
Ensemble Methods: More advanced models that combine multiple models like Decision Trees or Random Forest to improve overall accuracy [2, 3] [NUMBER: https://www.sciencedirect.com/science/article/pii/S2666764923000218 NUMBER: https://www.scirp.org/journal/paperinformation.aspx?paperid=120102].
These models are trained on historical loan data to identify patterns and relationships between borrower attributes and defaults. Once a model is trained, it can be used to score new loan applications and predict the likelihood of default.

Sources
Analytics Vidhya: https://www.analyticsvidhya.com/blog/2022/04/predicting-possible-loan-default-using-machine-learning/ Predicting Possible Loan Default Using Machine Learning
ScienceDirect: https://www.sciencedirect.com/science/article/pii/S2666764923000218 Explainable prediction of loan default based on machine ...
SCIRP: https://www.scirp.org/journal/paperinformation.aspx?paperid=120102 Machine Learning Approaches to Predict Loan Default



# Loan-Default-Prediction-Model 🚀

> A complete, explainable pipeline in Python to predict whether a borrower will default on a loan — from raw data and exploratory analysis through model training, evaluation and selection.

---

## Table of Contents
1. [Background & Motivation](#background--motivation)  
2. [Dataset](#dataset)  
3. [Exploratory Data Analysis](#exploratory-data-analysis)  
4. [Preprocessing & Feature Engineering](#preprocessing--feature-engineering)  
5. [Modeling Workflow](#modeling-workflow)  
6. [Evaluation & Results](#evaluation--results)  
7. [Usage Instructions](#usage-instructions)  
8. [Repository Structure](#repository-structure)  
9. [Next Steps & Ideas](#next-steps--ideas)  
10. [License & Contact](#license--contact)

---

## 📌 Background & Motivation

Loan default poses a significant financial risk to lenders. Effective risk assessment enables banks and fintech platforms to make informed approval decisions and adjust interest rates. This project demonstrates a reproducible machine learning pipeline—from raw loan data to deployment-ready predictive model—with transparent metrics and methodology.

---

## Dataset

⚙️ *Note: Replace this placeholder with your actual dataset name, source URL, and relevant details.*

- **Graph-like historical** loan data including borrower demographics, credit history, loan characteristics (e.g. amount, term, interest rate).
- **Target variable**: `default` flag—1 for defaulted, 0 for repaid.
- **Data size**: N ≈ _xx_, features: F ≈ _yy_.

*(This repo currently includes Jupyter notebooks for EDA and modeling; update dataset info accordingly.)*

---

## 📊 Exploratory Data Analysis (EDA)

The notebook **`Loan Default Prediction EDA.ipynb`** ($\approx$Notebook 1) contains:

- Distribution and imbalance checks.
- Outlier detection (IQR, log-scaling, z-score).
- Correlation heatmaps and feature target‐dependency plots.
- Insights flagged from features like income/debt ratio, employment duration, credit inquiries.

This guided feature cleaning and informed the modeling strategy.

---

## 🧠 Preprocessing & Feature Engineering

Steps executed include:

- Handling missing values (`.dropna()`, median/mean for numericals, mode for categoricals).
- Encoding categorical variables using **LabelEncoder** or custom ordinal mappings.
- Binning skewed continuous variables (e.g. **DEBTINC**, **CLAGE**, **LoanAmount**).
- Addressing class imbalance using **RandomOverSampler** or **SMOTE**.
- Feature scaling via **StandardScaler**, as applicable.

These preprocessing steps were integrated into both notebooks for consistency.

---

## 🔍 Modeling Workflow

The notebook **`Loan Defaulter Predictor with Models.ipynb`** (Notebook 2) oversaw:

1. Splitting into train and test sets (typically 70:30), with stratification on default.
2. Training and evaluating classifiers:
   - **Logistic Regression** – baseline interpretable model.
   - **Decision Tree Classifier** – fast interpretable hierarchy.
   - **Random Forest** – for ensemble robustness.
   - (**Optional**) **XGBoost** – for high performance.
3. Hyperparameter tuning via `GridSearchCV` / `RandomizedSearchCV`.
4. Comparing models using cross-validated **ROC-AUC**, **Recall**, **Precision**, **F1-score**, and **confusion matrix**.
5. Persisting the final model with `joblib.dump(...)` for reproducible use.

The chosen final model achieved strong predictive performance while balancing recall (to minimize false negatives) and overall accuracy.

---

## 📈 Evaluation & Results

- **Final model** (e.g. Random Forest or XGBoost) selected by highest ROC-AUC and F1-score.
- Confusion matrices and classification reports included.
- Feature importance visualized via bar charts.
- Threshold-based analysis shows trade-offs between false positives and negatives.

*(Insert actual scores here — e.g., ROC-AUC = 0.88, recall = 0.82, F1-score = 0.80.)*

---

## ⚙️ Usage Instructions

1. **Clone** this repo:
   ```bash
   git clone https://github.com/DragonGodMonarchMk/Loan-Default-Prediction-Model.git
   cd Loan-Default-Prediction-Model

Corporate Finance Institute: https://corporatefinanceinstitute.com/course/loan-default-prediction-with-machine-learning/ Loan Default Prediction with Machine Learning
GitHub: https://github.com/suhasmaddali/Predicting-Loan-Default-Using-Machine-Learning suhasmaddali/Predicting-Loan-Default-Using-Machine ...
California State University, Fullerton: https://scholarworks.calstate.edu/downloads/8k71nk666 Predicting Loan Defaults using Machine Learning Techniques



