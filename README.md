# Customer Churn Prediction Using Machine Learning

## Overview
A supervised machine-learning project that predicts whether a customer is likely to leave a service provider. It includes data preprocessing, model training and evaluation, saved scikit-learn artifacts, command-line prediction, and a Streamlit dashboard.

## Dataset
The project uses the IBM Telco Customer Churn dataset in `data/churn.csv`. The target is `Churn` (`No` = stay, `Yes` = leave). Features cover customer profile, subscribed services, contract details, and billing. The customer identifier is excluded from predictive features.

## Workflow
1. Load and validate the dataset.
2. Remove duplicate records and invalid target rows.
3. Convert the target to numeric labels and clean numeric-like fields.
4. Split the data into training and testing sets.
5. Impute and scale numeric features; impute and one-hot encode categorical features.
6. Train Logistic Regression and Random Forest.
7. Select the model by F1-score and save the fitted model and metadata.

## Reported evaluation
The supplied `models/metadata.json` records these held-out metrics:

| Model | F1 | Precision | Recall | Accuracy | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6136 | 0.5043 | 0.7834 | 0.7381 | 0.8415 |
| Random Forest | 0.6124 | 0.5361 | 0.7139 | 0.7601 | 0.8351 |

Logistic Regression was selected by F1-score. These figures are transcribed from the supplied metadata and have not been independently reproduced in this repository.

**Model compatibility:** the supplied pickle artifacts were serialized with scikit-learn 1.9.0 according to the project metadata. Use the pinned dependencies in `requirements.txt`; scikit-learn pickle artifacts may be incompatible across versions.

## Setup
Use Python 3.13, create a virtual environment, and install dependencies:

```bash
pip install -r requirements.txt
```

## Train
From the repository root:

```bash
python src/train.py
```

Optional target-column override:

```bash
python src/train.py --target-column Churn
```

## Command-line prediction
Example input:

```bash
echo "{\"tenure\":12,\"MonthlyCharges\":78.5,\"TotalCharges\":\"942.0\",\"Contract\":\"Month-to-month\"}" | python src/predict.py
```

## Streamlit dashboard
Launch the interactive application:

```bash
streamlit run app/app.py
```

The dashboard accepts a customer profile and displays a churn prediction, stay/leave probabilities, and risk information.

## Project structure
```text
.
├── app/app.py
├── data/churn.csv
├── models/
│   ├── encoder.pkl
│   ├── metadata.json
│   ├── model.pkl
│   └── scaler.pkl
├── src/
│   ├── __init__.py
│   ├── predict.py
│   ├── preprocess.py
│   └── train.py
├── utils/
│   ├── __init__.py
│   ├── config.py
│   └── logger.py
├── requirements.txt
└── README.md
```

## Team
- Ali Choukrane (23172510) — Team Leader
- Nabil Ouanou (23170934)
- Mohamed Idaissa (23170917)
- Mohammed Elyassaa (23170920)

## Notes
The source package was prepared without virtual environments, caches, temporary files, logs, or IDE configuration. Dataset and serialized model artifacts should be handled in accordance with their original source and licensing terms.
