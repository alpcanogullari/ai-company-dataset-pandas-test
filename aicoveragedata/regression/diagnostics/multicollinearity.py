import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Allow this file to run directly from the regression folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from aicoveragedata.regression.baseline import split_data
from aicoveragedata.regression.legacy_utils import load_regression_data


HIGH_CORRELATION_THRESHOLD = 0.80


def interpret_vif(value):
    if value < 5:
        return "acceptable"
    if value < 10:
        return "moderate multicollinearity"
    return "strong multicollinearity"


def calculate_vif(X):
    vif_rows = []

    for feature in X.columns:
        # Predict one feature using all other features.
        other_features = [column for column in X.columns if column != feature]
        model = LinearRegression()
        model.fit(X[other_features], X[feature])

        predicted_feature = model.predict(X[other_features])
        r2 = r2_score(X[feature], predicted_feature)
        vif = float("inf") if r2 >= 1 else 1 / (1 - r2)

        vif_rows.append(
            {
                "feature": feature,
                "vif": vif,
                "interpretation": interpret_vif(vif),
            }
        )

    return pd.DataFrame(vif_rows).sort_values("vif", ascending=False)


def high_correlation_pairs(X):
    correlation = X.corr()
    pairs = []

    for index, first_feature in enumerate(correlation.columns):
        for second_feature in correlation.columns[index + 1 :]:
            value = correlation.loc[first_feature, second_feature]
            if abs(value) >= HIGH_CORRELATION_THRESHOLD:
                pairs.append(
                    {
                        "feature_1": first_feature,
                        "feature_2": second_feature,
                        "correlation": value,
                    }
                )

    return pd.DataFrame(pairs).sort_values("correlation", key=abs, ascending=False)


def main():
    X, y, target_column, feature_columns = load_regression_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Multicollinearity Check")
    print("Rows used: training data only")
    print(f"Training rows: {len(X_train):,}")
    print(f"Target: {target_column}")
    print(f"Features: {', '.join(feature_columns)}")
    print()

    print("High correlation pairs")
    print(f"Threshold: abs(correlation) >= {HIGH_CORRELATION_THRESHOLD}")
    print(high_correlation_pairs(X_train).round(4).to_string(index=False))
    print()

    print("VIF scores")
    print(calculate_vif(X_train).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
