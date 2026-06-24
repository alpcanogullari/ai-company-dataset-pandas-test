import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
    )


def build_pipeline():
    return Pipeline(
        [
            ("scaler", RobustScaler()),
            ("model", LinearRegression()),
        ]
    )


def build_ridge_pipeline(alpha=10):
    return Pipeline(
        [
            ("scaler", RobustScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def train_model(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline


def interpret_score(name, value):
    if name == "R2":
        if value >= 0.80:
            return "strong fit"
        if value >= 0.50:
            return "moderate fit"
        if value >= 0.20:
            return "weak fit"
        return "poor fit"
    return "lower is better"


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    return {
        "r2": r2_score(y, y_pred),
        "mae": mean_absolute_error(y, y_pred),
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
    }


def print_metrics(title, metrics):
    print(title)
    print(f"R2: {metrics['r2']:.4f} ({interpret_score('R2', metrics['r2'])})")
    print(f"MAE: {metrics['mae']:.4f} ({interpret_score('MAE', metrics['mae'])})")
    print(f"RMSE: {metrics['rmse']:.4f} ({interpret_score('RMSE', metrics['rmse'])})")


def print_overfitting_check(train_metrics, test_metrics):
    difference = abs(train_metrics["r2"] - test_metrics["r2"])

    print()
    print("Overfitting check")
    print(f"Training R2: {train_metrics['r2']:.4f}")
    print(f"Test R2: {test_metrics['r2']:.4f}")
    print(f"Difference: {difference:.4f}")
    print("Result: no strong overfitting" if difference <= 0.05 else "Result: possible overfitting")


def get_coefficients(model, feature_columns):
    return pd.Series(
        model.named_steps["model"].coef_,
        index=feature_columns,
    ).sort_values(key=abs, ascending=False)


def print_coefficients(model, feature_columns):
    coefficients = get_coefficients(model, feature_columns)

    print()
    print("Most important coefficients:")
    print(coefficients.head(5))
