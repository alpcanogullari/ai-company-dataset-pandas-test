import html

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from .config import CORRELATION_EXCLUDE_COLUMNS, FEATURES, NEAR_ZERO_CORRELATION_LIMIT, TARGET

def polynomial_design(values, degree):
    values = np.asarray(values, dtype=float)
    return np.column_stack([values ** power for power in range(1, degree + 1)])


def univariate_test_r2(frame, feature, transform="raw", degree=1):
    paired = frame[[feature, TARGET]].dropna().copy()
    if len(paired) < 20 or paired[feature].nunique() < 3:
        return np.nan

    X_train, X_test, y_train, y_test = train_test_split(
        paired[[feature]],
        paired[TARGET],
        test_size=0.30,
        random_state=42,
    )

    train_values = X_train[feature].astype(float).to_numpy()
    test_values = X_test[feature].astype(float).to_numpy()
    if transform == "log1p":
        if np.nanmin(train_values) < 0 or np.nanmin(test_values) < 0:
            return np.nan
        train_values = np.log1p(train_values)
        test_values = np.log1p(test_values)

    center = float(np.nanmean(train_values))
    scale = float(np.nanstd(train_values)) or 1.0
    train_design = polynomial_design((train_values - center) / scale, degree)
    test_design = polynomial_design((test_values - center) / scale, degree)

    model = LinearRegression()
    model.fit(train_design, y_train)
    return r2_score(y_test, model.predict(test_design))


def nonlinear_signal_label(spearman_gap, nonlinear_lift):
    if nonlinear_lift >= 0.05 or spearman_gap >= 0.10:
        return "strong nonlinear signal"
    if nonlinear_lift >= 0.02 or spearman_gap >= 0.05:
        return "moderate nonlinear signal"
    return "mostly linear"


def nonlinear_shape_label(spearman_gap, quadratic_lift, log_lift):
    if log_lift >= 0.02 and log_lift >= quadratic_lift:
        return "log-shaped"
    if quadratic_lift >= 0.02:
        return "curved"
    if spearman_gap >= 0.05:
        return "monotonic nonlinear"
    return "linear enough"


def numeric_correlation_features(frame):
    return [
        column
        for column in frame.select_dtypes(include="number").columns
        if column not in CORRELATION_EXCLUDE_COLUMNS + [TARGET]
    ]


def build_nonlinear_correlation_checks(frame):
    rows = []
    correlation_features = numeric_correlation_features(frame)
    complete_df = frame[[TARGET] + correlation_features].dropna().copy()

    for feature in correlation_features:
        paired = complete_df[[feature, TARGET]].dropna()
        pearson = paired[feature].corr(paired[TARGET], method="pearson")
        spearman = paired[feature].corr(paired[TARGET], method="spearman")
        linear_r2 = univariate_test_r2(complete_df, feature, "raw", 1)
        quadratic_r2 = univariate_test_r2(complete_df, feature, "raw", 2)
        log_r2 = (
            univariate_test_r2(complete_df, feature, "log1p", 1)
            if paired[feature].min() >= 0
            else np.nan
        )
        spearman_gap = abs(spearman) - abs(pearson)
        quadratic_lift = quadratic_r2 - linear_r2
        log_lift = log_r2 - linear_r2 if not pd.isna(log_r2) else np.nan
        lift_values = [value for value in [quadratic_lift, log_lift] if not pd.isna(value)]
        nonlinear_lift = max(lift_values) if lift_values else np.nan

        rows.append(
            {
                "Feature": feature,
                "Model Feature": "Yes" if feature in FEATURES else "No",
                "Rows": len(paired),
                "Pearson Correlation": pearson,
                "Spearman Correlation": spearman,
                "Abs Spearman Minus Pearson": spearman_gap,
                "Linear Test R Square": linear_r2,
                "Quadratic Test R Square": quadratic_r2,
                "Log Test R Square": log_r2,
                "Best Nonlinear Lift": nonlinear_lift,
                "Near Zero Pearson": "Yes"
                if abs(pearson) <= NEAR_ZERO_CORRELATION_LIMIT
                else "No",
                "Shape": nonlinear_shape_label(spearman_gap, quadratic_lift, log_lift),
                "Signal": nonlinear_signal_label(spearman_gap, nonlinear_lift),
            }
        )

    return pd.DataFrame(rows).sort_values("Best Nonlinear Lift", ascending=False)


def nonlinear_correlation_note(nonlinear_checks):
    weakest = nonlinear_checks.sort_values("Pearson Correlation", key=abs).iloc[0]
    near_zero_rows = nonlinear_checks[
        nonlinear_checks["Pearson Correlation"].abs() <= NEAR_ZERO_CORRELATION_LIMIT
    ]
    top_gap = nonlinear_checks.sort_values("Abs Spearman Minus Pearson", ascending=False).iloc[0]
    top_lift = nonlinear_checks.sort_values("Best Nonlinear Lift", ascending=False).iloc[0]
    near_zero_note = (
        f"No numeric predictor has near-zero Pearson correlation with {TARGET}; "
        f"weakest is {html.escape(weakest['Feature'])} at {float(weakest['Pearson Correlation']):.4f}."
        if near_zero_rows.empty
        else f"{len(near_zero_rows)} numeric predictor(s) have near-zero Pearson correlation with {TARGET}."
    )
    return (
        "Pearson checks straight-line correlation. Spearman, quadratic, and log checks show whether the relationship bends. "
        f"{near_zero_note} "
        f"Example rank gap: {html.escape(top_gap['Feature'])} has Spearman-Pearson gap {float(top_gap['Abs Spearman Minus Pearson']):.4f}. "
        f"Best curve lift: {html.escape(top_lift['Feature'])} gains {float(top_lift['Best Nonlinear Lift']):.4f} test R Square."
    )


def empty_near_zero_pairwise_frame():
    return pd.DataFrame(
        columns=[
            "Feature 1",
            "Feature 2",
            "Pearson Correlation",
            "Spearman Correlation",
            "Abs Spearman Minus Pearson",
            "Linear Test R Square",
            "Quadratic Test R Square",
            "Best Nonlinear Lift",
            "Signal",
        ]
    )


def build_near_zero_pairwise_nonlinearity_checks(frame):
    numeric_columns = [
        column
        for column in frame.select_dtypes(include="number").columns
        if column not in CORRELATION_EXCLUDE_COLUMNS
    ]
    rows = []

    for first_index, first_feature in enumerate(numeric_columns):
        for second_feature in numeric_columns[first_index + 1 :]:
            paired = frame[[first_feature, second_feature]].dropna()
            pearson = paired[first_feature].corr(paired[second_feature], method="pearson")
            if pd.isna(pearson) or abs(pearson) > NEAR_ZERO_CORRELATION_LIMIT:
                continue

            spearman = paired[first_feature].corr(paired[second_feature], method="spearman")
            linear_r2 = univariate_pair_test_r2(paired, first_feature, second_feature, 1)
            quadratic_r2 = univariate_pair_test_r2(paired, first_feature, second_feature, 2)
            nonlinear_lift = quadratic_r2 - linear_r2
            spearman_gap = abs(spearman) - abs(pearson)
            rows.append(
                {
                    "Feature 1": first_feature,
                    "Feature 2": second_feature,
                    "Pearson Correlation": pearson,
                    "Spearman Correlation": spearman,
                    "Abs Spearman Minus Pearson": spearman_gap,
                    "Linear Test R Square": linear_r2,
                    "Quadratic Test R Square": quadratic_r2,
                    "Best Nonlinear Lift": nonlinear_lift,
                    "Signal": nonlinear_signal_label(spearman_gap, nonlinear_lift),
                }
            )

    if not rows:
        return empty_near_zero_pairwise_frame()
    return pd.DataFrame(rows).sort_values("Best Nonlinear Lift", ascending=False)


def univariate_pair_test_r2(frame, x_column, y_column, degree=1):
    if len(frame) < 20 or frame[x_column].nunique() < 3 or frame[y_column].nunique() < 3:
        return np.nan

    X_train, X_test, y_train, y_test = train_test_split(
        frame[[x_column]],
        frame[y_column],
        test_size=0.30,
        random_state=42,
    )
    train_values = X_train[x_column].astype(float).to_numpy()
    test_values = X_test[x_column].astype(float).to_numpy()
    center = float(np.nanmean(train_values))
    scale = float(np.nanstd(train_values)) or 1.0
    train_design = polynomial_design((train_values - center) / scale, degree)
    test_design = polynomial_design((test_values - center) / scale, degree)

    model = LinearRegression()
    model.fit(train_design, y_train)
    return r2_score(y_test, model.predict(test_design))
