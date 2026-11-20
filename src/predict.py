"""Inference module for customer churn predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from json import JSONDecodeError
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import METADATA_PATH, MODEL_PATH


class ChurnPredictor:
    """Load the trained sklearn Pipeline and make raw-input predictions."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        metadata_path: Path = METADATA_PATH,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {model_path}. Run `python src/train.py` "
                "after placing a real dataset at data/churn.csv."
            )
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Model metadata not found at {metadata_path}. Re-run training to create it."
            )

        self.pipeline = joblib.load(model_path)
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.feature_columns = self.metadata["feature_columns"]
        self.numeric_features = set(self.metadata.get("numeric_features", []))
        self.categorical_features = set(self.metadata.get("categorical_features", []))
        self.threshold = float(self.metadata.get("prediction_threshold", 0.5))

    def _to_frame(self, raw_input: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
        if isinstance(raw_input, pd.DataFrame):
            frame = raw_input.copy()
        elif isinstance(raw_input, dict):
            frame = pd.DataFrame([raw_input])
        elif isinstance(raw_input, list):
            frame = pd.DataFrame(raw_input)
        else:
            raise TypeError("raw_input must be a dict, list of dicts, or pandas DataFrame.")

        for column in self.feature_columns:
            if column not in frame.columns:
                frame[column] = np.nan

        frame = frame[self.feature_columns].copy()
        for column in self.numeric_features:
            if column in frame.columns:
                cleaned = frame[column].replace(r"^\s*$", np.nan, regex=True)
                frame[column] = pd.to_numeric(cleaned, errors="coerce")

        for column in self.categorical_features:
            if column in frame.columns:
                frame[column] = frame[column].replace(r"^\s*$", np.nan, regex=True)

        return frame

    def predict(self, raw_input: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame) -> list[dict[str, Any]]:
        """Return Stay/Leave predictions and probability scores."""
        frame = self._to_frame(raw_input)
        probabilities = self.pipeline.predict_proba(frame)
        classifier = self.pipeline.named_steps["classifier"]
        positive_index = list(classifier.classes_).index(1)

        leave_probabilities = probabilities[:, positive_index]
        stay_probabilities = 1.0 - leave_probabilities

        results: list[dict[str, Any]] = []
        for stay_probability, leave_probability in zip(stay_probabilities, leave_probabilities):
            prediction = "Leave" if leave_probability >= self.threshold else "Stay"
            confidence = max(float(stay_probability), float(leave_probability))
            results.append(
                {
                    "prediction": prediction,
                    "stay_probability": round(float(stay_probability), 4),
                    "leave_probability": round(float(leave_probability), 4),
                    "probability_score": round(float(leave_probability), 4),
                    "confidence": round(confidence, 4),
                }
            )
        return results

    def predict_one(self, raw_input: dict[str, Any] | pd.DataFrame) -> dict[str, Any]:
        """Return the first prediction for a single customer."""
        return self.predict(raw_input)[0]


def predict_churn(raw_input: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame) -> list[dict[str, Any]]:
    """Convenience function for batch predictions."""
    predictor = ChurnPredictor()
    return predictor.predict(raw_input)


def load_cli_payload(argv: list[str]) -> Any:
    """Load JSON payload from the first argument or stdin."""
    payload_text = argv[1] if len(argv) > 1 else sys.stdin.read()
    payload_text = payload_text.lstrip("\ufeff").strip()

    if not payload_text:
        raise ValueError(
            "No prediction payload supplied. Pass JSON as the first argument or pipe JSON to stdin."
        )

    try:
        return json.loads(payload_text)
    except JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON prediction payload: {exc}") from exc


if __name__ == "__main__":
    try:
        payload = load_cli_payload(sys.argv)
        print(json.dumps(predict_churn(payload), indent=2))
    except Exception as exc:
        print(f"Prediction failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
