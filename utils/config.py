"""Central project configuration."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

DATA_PATH = DATA_DIR / "churn.csv"
MODEL_PATH = MODEL_DIR / "model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
ENCODER_PATH = MODEL_DIR / "encoder.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"

TARGET_COLUMN = os.getenv("CHURN_TARGET_COLUMN")
TARGET_COLUMN_CANDIDATES = (
    "Churn",
    "churn",
    "Exited",
    "exited",
    "Attrition",
    "attrition",
    "Attrition_Flag",
    "Customer_Status",
    "customer_status",
    "target",
    "label",
)

POSITIVE_TARGET_VALUES = {
    "1",
    "yes",
    "true",
    "leave",
    "left",
    "churn",
    "churned",
    "exited",
    "attrited customer",
}

NEGATIVE_TARGET_VALUES = {
    "0",
    "no",
    "false",
    "stay",
    "stayed",
    "active",
    "existing customer",
    "retained",
}

IDENTIFIER_COLUMN_HINTS = (
    "id",
    "customerid",
    "customer_id",
    "accountid",
    "account_id",
    "clientid",
    "client_id",
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
HIGH_RISK_THRESHOLD = 0.70
