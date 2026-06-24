import numpy as np
import pandas as pd

from .config import FEATURES
from .formatting import add_error_percentages

def skew_direction(value):
    if pd.isna(value):
        return "Not available"
    if value > 0:
        return "Right-skewed"
    if value < 0:
        return "Left-skewed"
    return "Symmetric"


def skew_strength(value):
    if pd.isna(value):
        return "Not available"
    magnitude = abs(value)
    if magnitude < 0.5:
        return "Low"
    if magnitude < 1:
        return "Moderate"
    return "High"


def build_variable_skewness_table(predictions):
    test_data = add_error_percentages(predictions)
    sources = [
        ("Test feature", test_data[FEATURES]),
        (
            "Test target/error",
            test_data[
                [
                    "Actual Revenue Impact",
                    "Predicted Revenue Impact",
                    "Error",
                    "Error Percentage",
                ]
            ],
        ),
    ]
    rows = []

    for scope, frame in sources:
        for column in frame.columns:
            series = pd.to_numeric(frame[column], errors="coerce").dropna()
            value = series.skew() if len(series) > 2 else np.nan
            rows.append(
                {
                    "Variable": column,
                    "Scope": scope,
                    "Rows": len(series),
                    "P1": series.quantile(0.01),
                    "P10": series.quantile(0.10),
                    "P25": series.quantile(0.25),
                    "Mean": series.mean(),
                    "P50": series.quantile(0.50),
                    "Median": series.median(),
                    "P75": series.quantile(0.75),
                    "P90": series.quantile(0.90),
                    "P99": series.quantile(0.99),
                    "Std Dev": series.std(),
                    "Min": series.min(),
                    "Max": series.max(),
                    "Skewness": value,
                    "Skew Direction": skew_direction(value),
                    "Skew Strength": skew_strength(value),
                }
            )

    return pd.DataFrame(rows)


def build_feature_profile(frame):
    rows = []
    for feature in FEATURES:
        series = pd.to_numeric(frame[feature], errors="coerce").dropna()
        skewness_value = series.skew() if len(series) > 2 else np.nan
        rows.append(
            {
                "Variable": feature,
                "Rows": len(series),
                "P1": series.quantile(0.01),
                "P10": series.quantile(0.10),
                "P25": series.quantile(0.25),
                "Mean": series.mean(),
                "P50": series.quantile(0.50),
                "P75": series.quantile(0.75),
                "P90": series.quantile(0.90),
                "P99": series.quantile(0.99),
                "Std Dev": series.std(),
                "Min": series.min(),
                "Max": series.max(),
                "Skewness": skewness_value,
                "Skew Direction": skew_direction(skewness_value),
                "Skew Strength": skew_strength(skewness_value),
            }
        )
    return pd.DataFrame(rows)
