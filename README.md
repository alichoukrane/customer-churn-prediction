# Customer Churn Prediction Using Machine Learning

## Project Overview

This project predicts whether a customer is likely to leave a service provider using supervised machine learning. It includes a complete workflow for loading a real customer churn dataset, preprocessing the data, training candidate classification models, saving the best model, and running real-time predictions through a Streamlit web application.

The system is designed as a practical machine learning project with reusable source code, saved model artifacts, and a simple dashboard for entering customer information and viewing churn risk.

## Dataset Description

The project uses a customer churn dataset stored at:

```text
data/churn.csv
```

The included dataset is based on the public IBM Telco Customer Churn dataset. It contains customer account, service, and billing information, along with the target column:

```text
Churn
```

The target is binary:

- `No`: the customer stayed
- `Yes`: the customer churned

Main feature groups include:

- Customer profile: gender, senior citizen status, partner, dependents, tenure
- Services: phone service, internet service, online security, backup, device protection, technical support, streaming services
- Billing: contract type, paperless billing, payment method, monthly charges, total charges

The `customerID` column is removed during preprocessing because it is an identifier and should not be used as a predictive feature.

## Machine Learning Workflow

The training pipeline performs the following steps:

1. Loads `data/churn.csv`.
2. Validates that the dataset is present and not empty.
3. Infers or validates the churn target column.
4. Removes duplicate records.
5. Removes rows with missing or invalid target values.
6. Converts the target into numeric labels:
   - `0 = Stay`
   - `1 = Leave`
7. Converts numeric-like text columns, such as `TotalCharges`, into numeric values.
8. Removes identifier-like columns.
9. Splits the data into training and testing sets.
10. Applies preprocessing:
    - Numeric features: median imputation and standard scaling
    - Categorical features: most-frequent imputation and one-hot encoding
11. Trains Logistic Regression and Random Forest models.
12. Selects the best model using F1-score.
13. Saves the trained pipeline and metadata in the `models/` folder.

## Installation

Create and activate a Python environment, then install the required dependencies:

```bash
pip install -r requirements.txt
```

Recommended Python version:

```text
Python 3.13
```

## Training Instructions

From the project root folder, run:

```bash
python src/train.py
```

Optional target override:

```bash
python src/train.py --target-column Churn
```

After training, the following files are created or updated in `models/`:

```text
models/model.pkl
models/scaler.pkl
models/encoder.pkl
models/metadata.json
```

## Prediction From Command Line

A sample command-line prediction can be run with:

```bash
echo "{\"tenure\":12,\"MonthlyCharges\":78.5,\"TotalCharges\":\"942.0\",\"Contract\":\"Month-to-month\"}" | python src/predict.py
```

The prediction output includes:

- Final prediction: `Stay` or `Leave`
- Stay probability
- Leave probability
- Model confidence

## Running The Streamlit Application

To launch the web dashboard, run:

```bash
streamlit run app/app.py
```

The application provides:

- A customer profile input form
- Real-time churn prediction
- Stay and leave probability scores
- A probability visualization
- A high-risk warning when churn probability is high
- Model information and evaluation metrics in the sidebar

## Project Structure

```text
Customer_Churn_Prediction_Source_Code/
├── app/
│   └── app.py
├── data/
│   └── churn.csv
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

## Main Files

- `src/preprocess.py`: data loading, target handling, feature cleaning, preprocessing pipeline, and metadata creation
- `src/train.py`: model training, evaluation, model selection, and artifact saving
- `src/predict.py`: loading the saved model and generating churn predictions
- `app/app.py`: Streamlit dashboard for interactive customer churn prediction
- `utils/config.py`: project paths and constants
- `utils/logger.py`: logging setup

## Team Information

- ALI CHOUKRANE (23172510) - Team Leader
- NABIL OUANOU (23170934)
- Mohamed Idaissa (23170917)
- Mohammed Elyassaa (23170920)

## Submission Notes

This submission package excludes virtual environments, cache folders, Git metadata, temporary files, logs, and IDE configuration files. It contains only the files required to install, train, predict, and run the Streamlit application.
