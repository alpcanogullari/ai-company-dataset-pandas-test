import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .config import DATA_PATH, FEATURES, TARGET
from .formatting import adjusted_r2, error_direction
from .skew_transform import apply_skew_transform, build_skew_transform_audit, build_skew_transform_plan

def build_score_audit_table(raw_rows, model_rows, train_index, test_index, metrics):
    overlap = len(set(train_index).intersection(set(test_index)))
    r2_gap = abs(metrics["train_r2"] - metrics["test_r2"])
    fit_status = "Stable" if r2_gap <= 0.05 else "Review"

    return pd.DataFrame(
        [
            {"Check": "Raw rows", "Value": f"{raw_rows:,}", "Result": "source file loaded"},
            {"Check": "Rows used", "Value": f"{model_rows:,}", "Result": "after dropping missing model fields"},
            {"Check": "Rows dropped", "Value": f"{raw_rows - model_rows:,}", "Result": "missing required fields"},
            {"Check": "Training rows", "Value": f"{metrics['train_rows']:,}", "Result": "70% split"},
            {"Check": "Test rows", "Value": f"{metrics['test_rows']:,}", "Result": "30% split"},
            {"Check": "Train/test overlap", "Value": f"{overlap:,}", "Result": "pass" if overlap == 0 else "fail"},
            {"Check": "Training R Square", "Value": f"{metrics['train_r2']:.4f}", "Result": "moderate fit"},
            {"Check": "Test R Square", "Value": f"{metrics['test_r2']:.4f}", "Result": "moderate fit"},
            {"Check": "R Square gap", "Value": f"{r2_gap:.4f}", "Result": fit_status},
            {"Check": "Pipeline", "Value": "Skew transform -> RobustScaler -> LinearRegression", "Result": "fitted on training rows"},
        ]
    )


def build_metrics(y_train, train_pred, y_test, test_pred, feature_count):
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    return {
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_adj_r2": adjusted_r2(train_r2, len(y_train), feature_count),
        "test_adj_r2": adjusted_r2(test_r2, len(y_test), feature_count),
        "train_mae": mean_absolute_error(y_train, train_pred),
        "test_mae": mean_absolute_error(y_test, test_pred),
        "train_rmse": np.sqrt(mean_squared_error(y_train, train_pred)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
        "train_rows": len(y_train),
        "test_rows": len(y_test),
    }


def metrics_table(metrics):
    return pd.DataFrame(
        [
            {"Metric": "R Square", "Training Data": metrics["train_r2"], "Test Data": metrics["test_r2"]},
            {"Metric": "Adjusted R Square", "Training Data": metrics["train_adj_r2"], "Test Data": metrics["test_adj_r2"]},
            {"Metric": "MAE", "Training Data": metrics["train_mae"], "Test Data": metrics["test_mae"]},
            {"Metric": "RMSE", "Training Data": metrics["train_rmse"], "Test Data": metrics["test_rmse"]},
            {"Metric": "Observations", "Training Data": metrics["train_rows"], "Test Data": metrics["test_rows"]},
        ]
    )


def pipeline_audit_table(pipeline, feature_columns, metrics, split_description):
    return pd.DataFrame(
        [
            {"Check": "Target", "Result": TARGET},
            {"Check": "Features", "Result": ", ".join(feature_columns)},
            {"Check": "Split", "Result": split_description},
            {"Check": "Training Rows", "Result": f"{metrics['train_rows']:,}"},
            {"Check": "Test Rows", "Result": f"{metrics['test_rows']:,}"},
            {"Check": "Pipeline Steps", "Result": "Skew transform -> RobustScaler -> LinearRegression"},
            {"Check": "Scaler Fitted", "Result": "yes" if hasattr(pipeline.named_steps["scaler"], "center_") else "no"},
            {"Check": "Model Fitted", "Result": "yes" if hasattr(pipeline.named_steps["model"], "coef_") else "no"},
        ]
    )


def fit_model():
    df = pd.read_csv(DATA_PATH)
    model_df = df[["company_id", "year"] + FEATURES + [TARGET]].dropna().copy()
    X = model_df[FEATURES]
    y = model_df[TARGET]

    raw_X_train, raw_X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
    )
    skew_plan = build_skew_transform_plan(raw_X_train)
    X_train = apply_skew_transform(raw_X_train, skew_plan)
    X_test = apply_skew_transform(raw_X_test, skew_plan)
    transformed_model_df = model_df.copy()
    transformed_model_df[FEATURES] = transformed_model_df[FEATURES].astype(float)
    transformed_model_df.loc[X_train.index, FEATURES] = X_train
    transformed_model_df.loc[X_test.index, FEATURES] = X_test
    skew_audit = build_skew_transform_audit(raw_X_train, X_train, raw_X_test, X_test, skew_plan)

    pipeline = Pipeline(
        [
            ("scaler", RobustScaler()),
            ("model", LinearRegression()),
        ]
    )
    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)

    train_predictions = X_train.copy()
    train_predictions["Case ID"] = X_train.index
    train_predictions["Actual Revenue Impact"] = y_train
    train_predictions["Linear Regression Prediction"] = train_pred
    train_predictions["Error"] = y_train - train_pred
    train_predictions["Error Direction"] = train_predictions["Error"].map(error_direction)
    train_predictions["Absolute Error"] = train_predictions["Error"].abs()
    train_predictions["year"] = model_df.loc[X_train.index, "year"]
    train_predictions = train_predictions.sort_values("Case ID")

    predictions = X_test.copy()
    predictions["Case ID"] = X_test.index
    predictions["Actual Revenue Impact"] = y_test
    predictions["Predicted Revenue Impact"] = test_pred
    predictions["Linear Regression Prediction"] = test_pred
    predictions["Error"] = y_test - test_pred
    predictions["Error Direction"] = predictions["Error"].map(error_direction)
    predictions["Absolute Error"] = predictions["Error"].abs()
    predictions["year"] = model_df.loc[X_test.index, "year"]

    coefficients = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Coefficient": pipeline.named_steps["model"].coef_,
        }
    )
    coefficients["Feature Space"] = "Skew transformed"
    coefficients["Abs Coefficient"] = coefficients["Coefficient"].abs()
    coefficients["Direction"] = np.where(coefficients["Coefficient"] >= 0, "Positive", "Negative")
    coefficients = coefficients.sort_values("Abs Coefficient", ascending=False)
    intercept = pd.DataFrame(
        [
            {
                "Feature": "Intercept",
                "Coefficient": pipeline.named_steps["model"].intercept_,
                "Abs Coefficient": abs(pipeline.named_steps["model"].intercept_),
                "Direction": "Positive",
            }
        ]
    )
    coefficients = pd.concat([intercept, coefficients], ignore_index=True)

    test_errors = y_test - test_pred
    ss_res = float(np.sum(test_errors ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
    ss_reg = ss_tot - ss_res
    df_reg = len(FEATURES)
    df_res = len(y_test) - df_reg - 1
    ms_reg = ss_reg / df_reg
    ms_res = ss_res / df_res
    anova = pd.DataFrame(
        [
            {"Source": "Regression", "df": df_reg, "SS": ss_reg, "MS": ms_reg, "F": ms_reg / ms_res},
            {"Source": "Residual", "df": df_res, "SS": ss_res, "MS": ms_res, "F": ""},
            {"Source": "Total", "df": len(y_test) - 1, "SS": ss_tot, "MS": "", "F": ""},
        ]
    )

    metrics = build_metrics(y_train, train_pred, y_test, test_pred, len(FEATURES))
    score_audit = build_score_audit_table(
        len(df),
        len(model_df),
        raw_X_train.index,
        raw_X_test.index,
        metrics,
    )
    score_audit.loc[len(score_audit)] = {
        "Check": "Skew transform",
        "Value": "fit on 70% training rows",
        "Result": "applied to train and test rows",
    }
    score_audit.loc[len(score_audit)] = {
        "Check": "Feature space",
        "Value": "transformed independent variables",
        "Result": "target stays revenue_impact",
    }

    return (
        pipeline,
        transformed_model_df,
        train_predictions.drop(columns=["Absolute Error"]),
        predictions,
        coefficients,
        anova,
        metrics,
        score_audit,
        skew_audit,
    )
