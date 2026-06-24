import html

import numpy as np
import pandas as pd

from .config import DOWNLOAD_DIR, FEATURES
from .formatting import add_error_percentages, directional_outliers, number, percent

def export_tables(predictions, coefficients):
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    enriched = add_error_percentages(predictions)
    export_predictions = enriched[
        ["year"]
        + FEATURES
        + [
            "Case ID",
            "Actual Revenue Impact",
            "Predicted Revenue Impact",
            "Error",
            "Absolute Error",
            "Error Percentage",
            "Absolute Error Percentage",
            "Error Direction",
        ]
    ].copy().sort_values("Case ID")
    largest = export_predictions.loc[enriched.nlargest(5, "Absolute Error").index]
    top_under, top_over = directional_outliers(predictions)
    outlier_columns = (
        ["Error Direction", "Outlier Rank", "year"]
        + FEATURES
        + [
            "Case ID",
            "Actual Revenue Impact",
            "Predicted Revenue Impact",
            "Error",
            "Absolute Error",
            "Error Percentage",
            "Absolute Error Percentage",
        ]
    )
    top_under = top_under[outlier_columns].copy()
    top_over = top_over[outlier_columns].copy()
    error_percentages = pd.concat([top_under, top_over], ignore_index=True)

    coefficients.to_csv(DOWNLOAD_DIR / "selected_regression_coefficients.csv", index=False)
    export_predictions.to_csv(DOWNLOAD_DIR / "selected_regression_predictions.csv", index=False)
    largest.to_csv(DOWNLOAD_DIR / "selected_regression_largest_errors.csv", index=False)
    coefficients.to_csv(DOWNLOAD_DIR / "regression_coefficients.csv", index=False)
    export_predictions.head(500).to_csv(DOWNLOAD_DIR / "regression_predictions_sample.csv", index=False)
    largest.to_csv(DOWNLOAD_DIR / "regression_largest_error_cases.csv", index=False)
    top_under.to_csv(DOWNLOAD_DIR / "regression_top_underpredicted_error_cases.csv", index=False)
    top_over.to_csv(DOWNLOAD_DIR / "regression_top_overpredicted_error_cases.csv", index=False)
    error_percentages.to_csv(DOWNLOAD_DIR / "regression_directional_outlier_error_percentages.csv", index=False)

    return export_predictions, largest, top_under, top_over, error_percentages


def format_table(frame):
    formatted = frame.copy()
    for column in formatted.columns:
        if column in [
            "Actual Revenue Impact",
            "Predicted Revenue Impact",
            "Error",
            "Absolute Error",
            "Coefficient",
            "Abs Coefficient",
            "Current Coefficient",
            "Lag Coefficient",
            "Abs Lag Coefficient",
            "SS",
            "MS",
            "F",
            "P1",
            "P10",
            "P25",
            "Mean",
            "P50",
            "Median",
            "P75",
            "P90",
            "P99",
            "Std Dev",
            "Min",
            "Max",
            "Original Threshold",
            "Threshold",
            "Node Prediction",
            "Decision Tree Prediction",
            "Linear Regression Prediction",
            "XGBoost Prediction",
            "XGBoost Signal",
            "Stacked Regression Prediction",
            "Left Prediction",
            "Right Prediction",
            "Native Gain",
            "Minimum Native Gain",
            "Raw Training Skewness",
            "Raw Skewness",
            "Transformed Skewness",
            "Raw Mean",
            "Transformed Mean",
            "Raw P50",
            "Transformed P50",
            "Transform Parameter",
        ]:
            formatted[column] = formatted[column].map(lambda value: "" if value == "" else number(value))
        elif column in [
            "ai_investment_usd",
            "cost_savings",
            "lag_1_ai_investment_usd",
            "lag_1_cost_savings",
            "lag_1_revenue_impact",
        ]:
            formatted[column] = formatted[column].map(lambda value: f"{float(value):,.0f}")
        elif column in ["Case ID", "df", "Node", "Depth", "Left Rows", "Right Rows"]:
            formatted[column] = formatted[column].map(lambda value: "" if value == "" else f"{int(value):,}")
        elif column in ["Rows"]:
            formatted[column] = formatted[column].map(lambda value: f"{int(value):,}")
        elif column in ["Value", "Training Data", "Test Data"]:
            formatted[column] = formatted[column].map(
                lambda value: number(value)
                if isinstance(value, (int, float, np.integer, np.floating))
                else html.escape(str(value))
            )
        elif column in ["R Square Gain", "Node Variance Gain", "Gain Threshold", "Importance"]:
            formatted[column] = formatted[column].map(lambda value: "" if value == "" else f"{float(value) * 100:.2f}%")
        elif column in ["Error Percentage", "Absolute Error Percentage"]:
            formatted[column] = formatted[column].map(percent)
        elif pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: f"{float(value):.4f}")
    return formatted
