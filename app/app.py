"""Streamlit dashboard for real-time customer churn prediction."""

from __future__ import annotations

import json
from math import isfinite
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import seaborn as sns
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import ChurnPredictor
from utils.config import HIGH_RISK_THRESHOLD, METADATA_PATH, MODEL_PATH


PROBABILITY_COLORS = {"Stay": "#157f60", "Leave": "#c2410c"}
PROBABILITY_SUM_TOLERANCE = 0.02
FEATURE_GROUPS = {
    "Account": {
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
    },
    "Services": {
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    },
    "Billing": {
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
    },
}


st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --surface: #ffffff;
                --surface-muted: #f6f8fb;
                --line: #d9dee8;
                --text: #172033;
                --muted: #5c667a;
                --green: #157f60;
                --orange: #c2410c;
                --blue: #175cd3;
            }
            .main .block-container {
                padding-top: 1.25rem;
                padding-bottom: 2.25rem;
                max-width: 1320px;
            }
            h1 {
                letter-spacing: 0;
            }
            .app-header {
                border-bottom: 1px solid var(--line);
                padding-bottom: 1rem;
                margin-bottom: 1.1rem;
            }
            .header-row {
                display: flex;
                justify-content: space-between;
                gap: 1rem;
                align-items: flex-end;
                flex-wrap: wrap;
            }
            .header-kicker {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                padding: 0.28rem 0.66rem;
                border: 1px solid #bfd4ff;
                border-radius: 999px;
                background: #edf4ff;
                color: var(--blue);
                font-size: 0.78rem;
                font-weight: 750;
                margin-bottom: 0.45rem;
            }
            .header-title {
                margin: 0;
                color: var(--text);
                font-size: clamp(2rem, 4vw, 3.15rem);
                line-height: 1.04;
            }
            .header-copy {
                margin: 0.55rem 0 0 0;
                color: var(--muted);
                font-size: 1rem;
            }
            .model-chip-row {
                display: flex;
                gap: 0.5rem;
                flex-wrap: wrap;
                justify-content: flex-end;
            }
            .model-chip {
                border: 1px solid var(--line);
                background: var(--surface);
                border-radius: 999px;
                padding: 0.35rem 0.72rem;
                color: var(--text);
                font-size: 0.82rem;
                font-weight: 650;
            }
            [data-testid="stMetric"] {
                background: var(--surface);
                border: 1px solid #e6e8ef;
                border-radius: 8px;
                padding: 14px 16px;
                box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            }
            .section-title {
                font-size: 1.05rem;
                font-weight: 700;
                color: var(--text);
                margin: 0.25rem 0 0.75rem 0;
            }
            .subtle-caption {
                color: var(--muted);
                font-size: 0.88rem;
                margin: -0.25rem 0 0.9rem 0;
            }
            .result-panel {
                border: 1px solid var(--line);
                border-radius: 8px;
                background: var(--surface);
                padding: 1rem;
                margin-bottom: 1rem;
            }
            .result-label {
                color: var(--muted);
                font-size: 0.78rem;
                font-weight: 750;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                margin-bottom: 0.25rem;
            }
            .result-value {
                color: var(--text);
                font-size: 2.2rem;
                line-height: 1.05;
                font-weight: 800;
            }
            .result-note {
                color: var(--muted);
                font-size: 0.9rem;
                margin-top: 0.35rem;
            }
            .risk-track {
                height: 12px;
                border-radius: 999px;
                overflow: hidden;
                background: linear-gradient(90deg, #dff5ec 0%, #fef3c7 55%, #fee2d5 100%);
                border: 1px solid #e4e7ec;
                margin: 0.85rem 0 0.4rem 0;
            }
            .risk-fill {
                height: 100%;
                border-radius: 999px;
                background: #172033;
            }
            .risk-scale {
                display: flex;
                justify-content: space-between;
                color: var(--muted);
                font-size: 0.76rem;
            }
            .empty-state {
                border: 1px dashed #cdd5e2;
                border-radius: 8px;
                background: var(--surface-muted);
                padding: 1.1rem;
                color: var(--muted);
            }
            section[data-testid="stSidebar"] .stMarkdown p {
                margin-bottom: 0.35rem;
            }
            div[data-testid="stForm"] {
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 1rem;
                background: var(--surface);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_predictor() -> ChurnPredictor | None:
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return None
    return ChurnPredictor()


@st.cache_data(show_spinner=False)
def load_metadata() -> dict[str, Any] | None:
    if not METADATA_PATH.exists():
        return None
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def humanize(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").strip().title()


def format_model_name(name: Any) -> str:
    if not name:
        return "Unknown"
    return humanize(str(name))


def format_metric_name(name: str) -> str:
    return name.replace("_", " ").upper() if name == "roc_auc" else humanize(name)


def default_numeric_value(summary: dict[str, Any]) -> float:
    return float(summary.get("median", summary.get("min", 0.0)) or 0.0)


def get_feature_group(feature: str) -> str:
    for group, features in FEATURE_GROUPS.items():
        if feature in features:
            return group
    return "Other"


def render_header(metadata: dict[str, Any] | None) -> None:
    selected_model = format_model_name(metadata.get("selected_model")) if metadata else "Not trained"
    target_column = metadata.get("target_column", "unknown") if metadata else "unknown"
    feature_count = len(metadata.get("feature_columns", [])) if metadata else 0

    st.markdown(
        f"""
        <div class="app-header">
            <div class="header-row">
                <div>
                    <div class="header-kicker">Production Churn AI</div>
                    <h1 class="header-title">Customer Churn Prediction</h1>
                    <p class="header-copy">Score customer retention risk from the trained churn pipeline.</p>
                </div>
                <div class="model-chip-row">
                    <div class="model-chip">Model: {selected_model}</div>
                    <div class="model-chip">Target: {target_column}</div>
                    <div class="model-chip">{feature_count} features</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _coerce_probability(value: Any, field_name: str) -> float | None:
    """Convert UI probability values to finite 0..1 floats."""
    if value is None:
        return None

    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None

        is_percent = cleaned.endswith("%")
        if is_percent:
            cleaned = cleaned[:-1].strip()

        try:
            probability = float(cleaned)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be numeric; received {value!r}.") from exc

        if is_percent:
            probability /= 100.0
    else:
        try:
            probability = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be numeric; received {value!r}.") from exc

    if not isfinite(probability):
        return None

    if probability < 0.0 or probability > 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1; received {probability}.")

    return probability


def normalize_probability_pair(
    stay_probability: Any,
    leave_probability: Any,
) -> tuple[float, float]:
    """Validate and complete Stay/Leave probabilities for display."""
    stay_value = _coerce_probability(stay_probability, "stay_probability")
    leave_value = _coerce_probability(leave_probability, "leave_probability")

    if stay_value is None and leave_value is None:
        raise ValueError("At least one valid Stay or Leave probability is required.")
    if stay_value is None:
        stay_value = 1.0 - leave_value
    if leave_value is None:
        leave_value = 1.0 - stay_value

    probability_sum = stay_value + leave_value
    if abs(probability_sum - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise ValueError(
            "Stay and Leave probabilities must sum to approximately 1.0; "
            f"received {probability_sum:.4f}."
        )

    return stay_value, leave_value


def build_customer_form(metadata: dict[str, Any]) -> dict[str, Any] | None:
    feature_columns = metadata.get("feature_columns", [])
    numeric_features = set(metadata.get("numeric_features", []))
    categorical_features = set(metadata.get("categorical_features", []))
    numeric_summary = metadata.get("numeric_summary", {})
    categorical_levels = metadata.get("categorical_levels", {})
    grouped_features: dict[str, list[str]] = {"Account": [], "Services": [], "Billing": [], "Other": []}

    for feature in feature_columns:
        grouped_features[get_feature_group(feature)].append(feature)

    st.markdown('<div class="section-title">Customer Profile</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtle-caption">Inputs are organized by account, service, and billing signals.</div>',
        unsafe_allow_html=True,
    )
    with st.form("customer_churn_form"):
        customer_input: dict[str, Any] = {}
        visible_groups = [group for group, features in grouped_features.items() if features]
        tabs = st.tabs(visible_groups)

        for tab, group in zip(tabs, visible_groups):
            with tab:
                left, right = st.columns(2)
                for index, feature in enumerate(grouped_features[group]):
                    column = left if index % 2 == 0 else right
                    with column:
                        label = humanize(feature)
                        if feature in numeric_features:
                            summary = numeric_summary.get(feature, {})
                            customer_input[feature] = st.number_input(
                                label,
                                value=default_numeric_value(summary),
                                step=1.0,
                                format="%.4f",
                            )
                        elif feature in categorical_features:
                            options = categorical_levels.get(feature, [])
                            if 0 < len(options) <= 50:
                                customer_input[feature] = st.selectbox(label, options=options)
                            else:
                                customer_input[feature] = st.text_input(label, value="")
                        else:
                            customer_input[feature] = st.text_input(label, value="")

        submitted = st.form_submit_button("Run Churn Prediction", type="primary", width="stretch")
    return customer_input if submitted else None


def render_probability_chart(stay_probability: Any, leave_probability: Any) -> None:
    try:
        stay_value, leave_value = normalize_probability_pair(stay_probability, leave_probability)
    except ValueError as exc:
        st.error(f"Cannot render probability chart: {exc}")
        return

    probabilities = pd.DataFrame(
        {
            "Outcome": ["Stay", "Leave"],
            "Probability": [stay_value, leave_value],
        }
    )

    fig, ax = plt.subplots(figsize=(7, 3.0))
    bars = ax.barh(
        probabilities["Outcome"],
        probabilities["Probability"],
        color=[PROBABILITY_COLORS[outcome] for outcome in probabilities["Outcome"]],
        height=0.42,
    )

    ax.set_xlim(0, 1)
    ax.set_ylabel("")
    ax.set_xlabel("Probability")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, probabilities["Probability"]):
        label_inside_bar = value >= 0.18
        label_x = value - 0.035 if label_inside_bar else value + 0.025
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1%}",
            ha="right" if label_inside_bar else "left",
            va="center",
            color="white" if label_inside_bar else "#182230",
            fontweight="bold",
        )

    sns.despine(ax=ax)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def render_prediction(result: dict[str, Any]) -> None:
    try:
        stay_probability, leave_probability = normalize_probability_pair(
            result.get("stay_probability"),
            result.get("leave_probability"),
        )
    except ValueError as exc:
        st.error(f"Prediction result contains invalid probability values: {exc}")
        st.json(result)
        return

    confidence = max(stay_probability, leave_probability)
    prediction = result.get("prediction") or ("Leave" if leave_probability >= 0.5 else "Stay")
    risk_label = "High risk" if leave_probability >= HIGH_RISK_THRESHOLD else "Moderate risk" if prediction == "Leave" else "Low risk"
    risk_width = min(max(leave_probability * 100, 0), 100)

    st.markdown('<div class="section-title">Prediction Summary</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="result-panel">
            <div class="result-label">{risk_label}</div>
            <div class="result-value">{prediction}</div>
            <div class="result-note">Leave risk {leave_probability:.1%} with {confidence:.1%} model confidence.</div>
            <div class="risk-track">
                <div class="risk-fill" style="width: {risk_width:.1f}%;"></div>
            </div>
            <div class="risk-scale">
                <span>Stay leaning</span>
                <span>Retention priority</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_left, metric_middle, metric_right = st.columns(3)
    metric_left.metric("Stay", f"{stay_probability:.1%}")
    metric_middle.metric("Leave", f"{leave_probability:.1%}")
    metric_right.metric("Confidence", f"{confidence:.1%}")

    render_probability_chart(stay_probability, leave_probability)

    if leave_probability >= HIGH_RISK_THRESHOLD:
        st.error(
            "High churn risk detected. Prioritize retention outreach, review service issues, "
            "and consider a targeted offer."
        )
    elif prediction == "Leave":
        st.warning("Moderate churn risk. Monitor this account and consider proactive engagement.")
    else:
        st.success("Customer is currently predicted to stay. Keep tracking behavior and service signals.")


def render_sidebar(metadata: dict[str, Any] | None) -> None:
    st.sidebar.title("Model Operations")
    if metadata is None:
        st.sidebar.warning("No trained model artifacts found.")
        st.sidebar.code("python src/train.py")
        return

    st.sidebar.success("Model loaded")
    st.sidebar.markdown(f"Selected model: `{format_model_name(metadata.get('selected_model'))}`")
    st.sidebar.markdown(f"Target: `{metadata.get('target_column', 'unknown')}`")
    st.sidebar.markdown(f"Features: `{len(metadata.get('feature_columns', []))}`")

    metrics = metadata.get("metrics", {})
    selected = metadata.get("selected_model")
    if selected in metrics:
        st.sidebar.divider()
        st.sidebar.markdown("Best model metrics")
        metric_rows = [
            {"Metric": format_metric_name(metric), "Value": value}
            for metric, value in metrics[selected].items()
        ]
        st.sidebar.dataframe(metric_rows, hide_index=True, width="stretch")

    st.sidebar.divider()
    st.sidebar.markdown(f"High risk threshold: `{HIGH_RISK_THRESHOLD:.0%}`")


def main() -> None:
    inject_styles()
    metadata = load_metadata()
    predictor = load_predictor()
    render_sidebar(metadata)

    render_header(metadata)

    if predictor is None or metadata is None:
        st.info(
            "Place a real churn dataset at data/churn.csv, then train the model before using "
            "the dashboard."
        )
        st.code("python src/train.py", language="bash")
        return

    input_col, output_col = st.columns([1.05, 0.95], gap="large")
    with input_col:
        customer_input = build_customer_form(metadata)

    with output_col:
        if customer_input is None:
            st.markdown('<div class="section-title">Prediction Summary</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="empty-state">Complete the customer profile and run a prediction.</div>',
                unsafe_allow_html=True,
            )
        else:
            result = predictor.predict_one(customer_input)
            render_prediction(result)


if __name__ == "__main__":
    main()
