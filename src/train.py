"""Train and persist the production churn prediction pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import build_metadata, build_preprocessor, load_dataset, prepare_training_frame
from utils.config import (
    DATA_PATH,
    ENCODER_PATH,
    METADATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    RANDOM_STATE,
    SCALER_PATH,
    TEST_SIZE,
)
from utils.logger import get_logger


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the customer churn prediction system.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="Path to the real churn CSV dataset.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default=None,
        help="Optional target column override. Otherwise common churn names are inferred.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Holdout test split size.",
    )
    return parser.parse_args()


def build_candidate_models() -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }


def evaluate_model(pipeline: Pipeline, X_test, y_test) -> dict[str, float]:
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, list(pipeline.named_steps["classifier"].classes_).index(1)]

    metrics = {
        "f1": f1_score(y_test, predictions, zero_division=0),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "accuracy": accuracy_score(y_test, predictions),
    }

    if len(set(y_test)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_test, probabilities)

    return {key: round(float(value), 4) for key, value in metrics.items()}


def save_pipeline_artifacts(best_pipeline: Pipeline, metadata: dict[str, Any]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipeline, MODEL_PATH)

    preprocessor = best_pipeline.named_steps["preprocessor"]
    numeric_transformer = preprocessor.named_transformers_.get("numeric")
    categorical_transformer = preprocessor.named_transformers_.get("categorical")

    scaler = None
    if numeric_transformer is not None and numeric_transformer != "drop":
        scaler = numeric_transformer.named_steps.get("scaler")

    encoder = None
    if categorical_transformer is not None and categorical_transformer != "drop":
        encoder = categorical_transformer.named_steps.get("encoder")

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    metadata = {
        **metadata,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "model": str(MODEL_PATH),
            "scaler": str(SCALER_PATH),
            "encoder": str(ENCODER_PATH),
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def train() -> dict[str, Any]:
    args = parse_args()

    logger.info("Loading dataset from %s", args.data_path)
    raw_data = load_dataset(args.data_path)
    logger.info("Loaded dataset with shape %s", raw_data.shape)

    training_frame = prepare_training_frame(raw_data, args.target_column)
    X = training_frame.features
    y = training_frame.target

    logger.info("Target column: %s", training_frame.target_column)
    logger.info("Removed duplicate rows: %s", training_frame.duplicate_rows_removed)
    logger.info("Removed rows with invalid/missing target: %s", training_frame.target_rows_removed)
    logger.info("Dropped identifier columns: %s", training_frame.dropped_columns or "none")
    logger.info("Training feature count: %s", X.shape[1])
    logger.info("Missing feature values will be handled inside the sklearn Pipeline.")

    if y.nunique() != 2:
        raise ValueError("Training target must contain both Stay and Leave classes.")

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    results: dict[str, dict[str, float]] = {}
    fitted_pipelines: dict[str, Pipeline] = {}

    for model_name, estimator in build_candidate_models().items():
        logger.info("Training candidate model: %s", model_name)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        results[model_name] = metrics
        fitted_pipelines[model_name] = pipeline
        logger.info("%s metrics: %s", model_name, metrics)

    best_model_name = max(results, key=lambda name: results[name]["f1"])
    best_pipeline = fitted_pipelines[best_model_name]
    logger.info("Selected best model by F1-score: %s", best_model_name)

    metadata = build_metadata(training_frame, best_model_name, results)
    save_pipeline_artifacts(best_pipeline, metadata)

    logger.info("Saved best pipeline to %s", MODEL_PATH)
    logger.info("Saved scaler to %s", SCALER_PATH)
    logger.info("Saved encoder to %s", ENCODER_PATH)
    logger.info("Saved metadata to %s", METADATA_PATH)

    return {
        "selected_model": best_model_name,
        "metrics": results,
        "model_path": str(MODEL_PATH),
    }


if __name__ == "__main__":
    try:
        summary = train()
        print(json.dumps(summary, indent=2))
    except Exception as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
